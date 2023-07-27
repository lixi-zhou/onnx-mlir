#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h.inc"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Transform/ONNX/ShapeInference.hpp"
#include <string>
#include <deque>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AffineParall
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-for-parallel"

using namespace mlir;
using namespace mlir::affine;



namespace {

LogicalResult affineForLoopParallelize(AffineForOp forOp,
                                ArrayRef<LoopReduction> parallelReductions) {
  // LLVM_DEBUG(llvm::dbgs() << "[TEST] affineForLoopParallelize: Current Exam Ops: " << forOp.getOperationName() <<  " Loc " << forOp.getLoc() << "\n");
  unsigned numReductions = parallelReductions.size();
  if (numReductions != forOp.getNumIterOperands())
    return failure();

  Location loc = forOp.getLoc();
  OpBuilder outsideBuilder(forOp);
  AffineMap lowerBoundMap = forOp.getLowerBoundMap();
  ValueRange lowerBoundOperands = forOp.getLowerBoundOperands();
  AffineMap upperBoundMap = forOp.getUpperBoundMap();
  ValueRange upperBoundOperands = forOp.getUpperBoundOperands();

  auto reducedValues = llvm::to_vector<4>(llvm::map_range(
      parallelReductions, [](const LoopReduction &red) { return red.value; }));
  auto reductionKinds = llvm::to_vector<4>(llvm::map_range(
      parallelReductions, [](const LoopReduction &red) { return red.kind; }));
  AffineParallelOp newPloop = outsideBuilder.create<AffineParallelOp>(
      loc, ValueRange(reducedValues).getTypes(), reductionKinds,
      llvm::ArrayRef(lowerBoundMap), lowerBoundOperands,
      llvm::ArrayRef(upperBoundMap), upperBoundOperands,
      llvm::ArrayRef(forOp.getStep()));
  newPloop.getRegion().takeBody(forOp.getRegion());
  Operation *yieldOp = &newPloop.getBody()->back();

  SmallVector<Value> newResults;
  newResults.reserve(numReductions);
  for (unsigned i = 0; i < numReductions; ++i) {
    Value init = forOp.getIterOperands()[i];
    Operation *reductionOp = yieldOp->getOperand(i).getDefiningOp();
    assert(reductionOp && "yielded value is expected to be produced by an op");
    outsideBuilder.getInsertionBlock()->getOperations().splice(
        outsideBuilder.getInsertionPoint(), newPloop.getBody()->getOperations(),
        reductionOp);
    reductionOp->setOperands({init, newPloop->getResult(i)});
    forOp->getResult(i).replaceAllUsesWith(reductionOp->getResult(0));
  }

  unsigned numIVs = 1;
  yieldOp->setOperands(reducedValues);
  newPloop.getBody()->eraseArguments(numIVs, numReductions);

  forOp.erase();
  return success();
}



struct AffineParallOptions {
  // unsigned maxNested = -1u;
  unsigned maxNested = 1;
  bool parallelReductions = false;
};

template <typename DerivedT>
class AffineParallBase : public ::mlir::OperationPass<func::FuncOp> {
public:
  using Base = AffineParallBase;

  AffineParallBase() : ::mlir::OperationPass<func::FuncOp>(::mlir::TypeID::get<DerivedT>()) {}
  AffineParallBase(const AffineParallBase &other) : ::mlir::OperationPass<func::FuncOp>(other) {}

  /// Returns the command-line argument attached to this pass.
  ::llvm::StringRef getArgument() const override { return "affine-for-parallel-dev"; }

  ::llvm::StringRef getDescription() const override { return "Convert affine.for ops into 1-D affine.parallel"; }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("AffineParall");
  }
  ::llvm::StringRef getName() const override { return "AffineParall"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    
  }

  /// Explicitly declare the TypeID for this class. We declare an explicit private
  /// instantiation because Pass classes should only be visible by the current
  /// library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineParallBase<DerivedT>)

  AffineParallBase(const AffineParallOptions &options) : AffineParallBase() {
    maxNested = options.maxNested;
    parallelReductions = options.parallelReductions;
  }
protected:
  ::mlir::Pass::Option<unsigned> maxNested{*this, "max-nested", ::llvm::cl::desc("Maximum number of nested parallel loops to produce. Defaults to unlimited (UINT_MAX)."), ::llvm::cl::init(-1u)};
  ::mlir::Pass::Option<bool> parallelReductions{*this, "parallel-reductions", ::llvm::cl::desc("Whether to parallelize reduction loops. Defaults to false."), ::llvm::cl::init(false)};
private:
};



struct AffineParall
    : public AffineParallBase<AffineParall> {
  void runOnOperation() override;
};

struct ParallelizationCandidate {
  ParallelizationCandidate(AffineForOp l, SmallVector<LoopReduction> &&r)
      : loop(l), reductions(std::move(r)) {}

  /// The potentially parallelizable loop.
  AffineForOp loop;
  /// Desciprtors of reductions that can be parallelized in the loop.
  SmallVector<LoopReduction> reductions;
};

bool isLoopParallelMod(
    AffineForOp forOp, SmallVectorImpl<LoopReduction> *parallelReductions) {
  unsigned numIterArgs = forOp.getNumIterOperands();

  // Loop is not parallel if it has SSA loop-carried dependences and reduction
  // detection is not requested.
  if (numIterArgs > 0 && !parallelReductions)
    return false;
  
  // Find supported reductions of requested.
  if (parallelReductions) {
    getSupportedReductions(forOp, *parallelReductions);
    // Return later to allow for identifying all parallel reductions even if the
    // loop is not parallel.
    if (parallelReductions->size() != numIterArgs)
      return false;
  }

  LLVM_DEBUG(llvm::dbgs() << "[TEST] loop analysis Flag 3 " << forOp.getLoc() << "\n");
  // the following check will fail on our onnx-mlir transformation
  bool isLoopMemoryParallelFlag = isLoopMemoryParallel(forOp);
  LLVM_DEBUG(llvm::dbgs() << "[TEST] loop analysis Flag 4 " << isLoopMemoryParallelFlag << " " << forOp.getLoc() << "\n");
  // Check memory dependences.
  // FIXME CHECK WHETHER THIS IS NEEDED
  // return isLoopMemoryParallel(forOp);
  return true;
}

} // namespace





void AffineParall::runOnOperation() {
  func::FuncOp f = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "[TEST] AffineParall runOnOperation " << f.getName() << " Loc " << f.getLoc() << "\n");
  std::string debugmsg = std::string("[STDERR] Run on Operation ") + f.getName().str().c_str() + " \n";
  fprintf(stderr, "%s", debugmsg.data());
  // fprintf(stderr, (char*)"[STDERR] Run on Operation" + f.getName().str().c_str() + " \n");
  // std::cerr << "[STDERR] Run on Operation".strdup() + f.getName().str().c_str() + " \n";
  // The walker proceeds in pre-order to process the outer loops first
  // and control the number of outer parallel loops.
  std::vector<ParallelizationCandidate> parallelizableLoops;
  f.walk<WalkOrder::PreOrder>([&](AffineForOp loop) {
    // fprintf(stderr, "%s", loop.getLoc().dyn_cast<mlir::StringAttr>().getValue().str());
    // mlir::LocationAttr loopLoc = loop.getLoc().getLocationAttr();
    // std::string debugmsg = std::string("[STDERR] Run on Operation ") + loop.getOperationName().str() +  std::to_string(loop.getLoc().dyn_cast<mlir::FileLineColLoc>().getLine()) + "\n";
    std::string debugmsg = std::string("[STDERR] Run on Operation ") + loop.getOperationName().str() + "\n";
    fprintf(stderr, "%s", debugmsg.data());
    // LLVM_DEBUG(llvm::dbgs() << "[TEST] AffineParall parallelizableLoops analysis " << loop.getOperationName() << " Loc " << loop.getLoc() << "\n");
    SmallVector<LoopReduction> reductions;
    bool isLoopParallelAnalysisFlag = isLoopParallelMod(loop, parallelReductions ? &reductions : nullptr);
    // LLVM_DEBUG(llvm::dbgs() << "[TEST] AffineParall parallelizableLoops analysis Flag: " << isLoopParallelAnalysisFlag << " " << loop.getOperationName() << " Loc " << loop.getLoc() << "\n");
    // isLoopParallelAnalysisFlag = true;
    // if (strcmp(loop.getLoc().str(), 'loc("example.affine.mlir":271:5)')) {
    //   isLoopParallelAnalysisFlag = true;
    // }
    // LLVM_DEBUG(llvm::dbgs() << "[TEST] AffineParall parallelizableLoops analysis Flag[FIX]: " << isLoopParallelAnalysisFlag << " " << loop.getOperationName() << " Loc " << loop.getLoc() << "\n");
    bool toBeParallelizedFlag = false;
    // TODO use the flag to filter opertor for parallelism
    if (loop->hasAttr(llvm::StringRef("toBeParallelized"))){
      fprintf(stderr, "%s", "This has parallel flag \n");
      toBeParallelizedFlag = true;
    } else {
      fprintf(stderr, "%s", "This does not have parallel flag \n");
    }
    // toBeParallelizedFlag = true;
    // mlir::StringAttr toBeParallelzedFlag = loop->getAttr(llvm::StringRef("toBeParallelized")).cast<mlir::StringAttr>();
    if (isLoopParallelAnalysisFlag && toBeParallelizedFlag)
      parallelizableLoops.emplace_back(loop, std::move(reductions));
  });

  for (const ParallelizationCandidate &candidate : parallelizableLoops) {
    unsigned numParentParallelOps = 0;
    AffineForOp loop = candidate.loop;
    // LLVM_DEBUG(llvm::dbgs() << "[TEST] AffineParall runOnOperation inner loop " << loop.getOperationName() << " Loc " << loop.getLoc() << "\n");
    for (Operation *op = loop->getParentOp();
         op != nullptr && !op->hasTrait<OpTrait::AffineScope>();
         op = op->getParentOp()) {
      if (isa<AffineParallelOp>(op))
        ++numParentParallelOps;
        fprintf(stderr, "%s", "[Lx] isa AffineParallelOp check \n");
    }
    maxNested = 1;
    std::string msg2 = "[Lx] numParentParallelOps: " + std::to_string(numParentParallelOps) + " , " + std::to_string(maxNested) + "\n";
    fprintf(stderr, "%s", msg2.data());
    if (numParentParallelOps < maxNested) {
      if (failed(affineForLoopParallelize(loop, candidate.reductions))) {
        LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] failed to parallelize\n"
                                << loop);
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] too many nested loops\n"
                              << loop);
    }
  }
}


namespace onnx_mlir {

std::unique_ptr<OperationPass<func::FuncOp>> createAffineParallPass() {
  return std::make_unique<AffineParall>();
}

// std::unique_ptr<mlir::Pass> createAffineParallPass() {
//   return std::make_unique<AffineParall>();
// }

// std::unique_ptr<mlir::Pass> createONNXOpenMPPass() {
//   return std::make_unique<ONNXOpenMPPass>();
// }

} // namespace onnx_mlir
