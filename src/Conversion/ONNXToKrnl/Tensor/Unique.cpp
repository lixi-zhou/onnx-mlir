/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- Unique.cpp - Lowering Unique Op ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Unique Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "onnx-mlir/Runtime/OMTensor.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "llvm/Support/Debug.h"
using namespace mlir;

namespace onnx_mlir {

/// Emit function call to compute arg unique of a given MemRef along a given
/// axis. The first output MemRef has the same shape as the input MemRef but is
/// of IndexType. Shape of the second, third and fourth arguments depends on the
/// input options.
Value emitArgUnique(ConversionPatternRewriter &rewriter, Location loc,
    Value total, Value input, int64_t axis, int64_t sorted, Value Y,
    Value indices, Value inverse_indices, Value counts, bool count_only) {
  MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder> create(
      rewriter, loc);
  IndexExprScope scope(create.krnl);
  MemRefType inputMemRefType = input.getType().cast<MemRefType>();
  int64_t rank = inputMemRefType.getRank();
  assert(axis < rank && "axis is out of bound");
  LiteralIndexExpr zeroIE(0), oneIE(1);
  SmallVector<IndexExpr, 4> lbs(rank, zeroIE);
  SmallVector<IndexExpr, 4> ubs;
  create.krnlIE.getShapeAsDims(input, ubs);

  // Emit krnl.Call to call omTensorUnique API
  Type int_type = rewriter.getIntegerType(64);
  Value val_axis = create.math.constant(int_type, axis);
  Value val_sorted = create.math.constant(int_type, sorted);
  if (count_only) {
    SmallVector<Value, 4> operands = {total, input, val_axis, val_sorted};
    rewriter.create<KrnlCallOp>(loc, "omTensorUniqueCount", 1, operands);
  } else {
    SmallVector<Value, 8> operands = {total, Y, indices, inverse_indices,
        counts, input, val_axis, val_sorted};
    rewriter.create<KrnlCallOp>(loc, "omTensorUnique", 5, operands);
  }
  return total;
}

struct ONNXUniqueOpLowering : public ConversionPattern {
  ONNXUniqueOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXUniqueOp::getOperationName(), 1, ctx) {}

  ///
  /// Intermediate data are presented below for better understanding:
  ///
  /// there are 4 subtensors sliced along axis 1 of input_x (shape = (2, 4, 2)):
  /// A: [[1, 1], [1, 1]],
  ///    [[0, 1], [0, 1]],
  ///    [[2, 1], [2, 1]],
  ///    [[0, 1], [0, 1]].
  ///
  /// there are 3 unique subtensors:
  /// [[1, 1], [1, 1]],
  /// [[0, 1], [0, 1]],
  /// [[2, 1], [2, 1]].
  ///
  /// sorted unique subtensors:
  /// B: [[0, 1], [0, 1]],
  ///    [[1, 1], [1, 1]],
  ///    [[2, 1], [2, 1]].
  ///
  /// output_Y is constructed from B:
  /// [[[0. 1.], [1. 1.], [2. 1.]],
  /// [[0. 1.], [1. 1.], [2. 1.]]]
  ///
  /// output_indices is to map from B to A:
  /// [1, 0, 2]
  ///
  /// output_inverse_indices is to map from A to B:
  /// [1, 0, 2, 0]
  ///
  /// output_counts = [2 1 1]
  ///

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXTopKOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    ONNXUniqueOp uniqueOp = llvm::cast<ONNXUniqueOp>(op);
    Location loc = op->getLoc();
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope scope(create.krnl);
    ONNXUniqueOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    Value X = operandAdaptor.getX();
    ArrayRef<int64_t> xShape = getShape(X.getType());
    Type elementType = X.getType().cast<MemRefType>().getElementType();
    int64_t rank = create.krnlIE.getShapedTypeRank(X);
    int64_t sorted = operandAdaptor.getSorted();
    std::optional<int64_t> optionalAxis = uniqueOp.getAxis();
    //
    // Get axis value as a positive integer if axis attribute is specified.
    // Set "-1" if the axis attribute is not specified.
    //
    int64_t axis = -1;
    if (optionalAxis.has_value()) {
      axis = optionalAxis.value();
      axis = axis < 0 ? axis + rank : axis;
      assert(axis >= 0 && axis < rank && "axis is out of bound");
    }
    //
    // Emit a Unique call to get the outputs to count unique
    // slices(subtensors) of X along axis at first.
    //
    Type indexTy = rewriter.getIndexType();
    Value iZero = create.math.constantIndex(0);
    Value uniqueCount = create.mem.alloca(MemRefType::get({}, indexTy));
    create.krnl.store(iZero, uniqueCount, {});
    Value noneValue;
    emitArgUnique(rewriter, loc, uniqueCount, X, axis, /*sorted=*/sorted,
        noneValue, noneValue, noneValue, noneValue, /*count_only=*/true);
    //
    // Calculate shapes of output Tensors
    //
    Value total = create.krnl.load(uniqueCount, {});
    NonAffineIndexExpr totalDimExpr = DimIndexExpr(total);
    DimsExpr outputYDims;
    DimsExpr outputIndexDims;
    DimsExpr outputInverseIndexDims;
    if (axis < 0) {
      outputYDims.emplace_back(totalDimExpr);
      outputIndexDims.emplace_back(totalDimExpr);
      DimIndexExpr inputDimExpr = LiteralIndexExpr(xShape[0]);
      for (int64_t i = 1; i < rank; i++) {
        inputDimExpr = inputDimExpr * LiteralIndexExpr(xShape[i]);
      }
      outputInverseIndexDims.emplace_back(inputDimExpr);
    } else {
      for (int64_t i = 0; i < rank; i++) {
        DimIndexExpr tDimExpr = LiteralIndexExpr(xShape[i]);
        if (i == axis)
          tDimExpr = totalDimExpr;
        outputYDims.emplace_back(tDimExpr);
      }
      outputIndexDims.emplace_back(totalDimExpr);
      outputInverseIndexDims.emplace_back(LiteralIndexExpr(xShape[axis]));
    }
    //
    // Insert an allocation and deallocation for the outputs.
    //
    Value outputY;
    if (axis < 0) {
      MemRefType memrefType =
          MemRefType::get({ShapedType::kDynamic}, elementType);
      outputY = create.mem.alignedAlloc(memrefType, outputYDims);
    } else {
      ArrayRef<int64_t> xShape = getShape(X.getType());
      SmallVector<int64_t> yShape;
      for (int i = 0; i < rank; i++)
        yShape.emplace_back((i == axis) ? ShapedType::kDynamic : xShape[i]);
      MemRefType memrefType = MemRefType::get(yShape, elementType);
      outputY = create.mem.alignedAlloc(memrefType, outputYDims);
    }
    Type i64Type = rewriter.getI64Type();
    MemRefType memrefType = MemRefType::get({ShapedType::kDynamic}, i64Type);
    Value emptyMemref = create.mem.alignedAlloc(MemRefType::get({0}, i64Type));
    Value indices = isNoneValue(uniqueOp.getIndices())
                        ? emptyMemref
                        : create.mem.alignedAlloc(memrefType, outputIndexDims);
    Value inverseIndices =
        isNoneValue(uniqueOp.getInverseIndices())
            ? emptyMemref
            : create.mem.alignedAlloc(memrefType, outputInverseIndexDims);
    Value counts = isNoneValue(uniqueOp.getCounts())
                       ? emptyMemref
                       : create.mem.alignedAlloc(memrefType, outputIndexDims);
    //
    // Emit a Unique call to get the outputs
    //
    create.krnl.store(iZero, uniqueCount, {});
    emitArgUnique(rewriter, loc, uniqueCount, X, axis, /*sorted=*/sorted,
        outputY, indices, inverseIndices, counts, /*count_only=*/false);
    if (isNoneValue(indices))
      indices = noneValue;
    if (isNoneValue(inverseIndices))
      inverseIndices = noneValue;
    if (isNoneValue(counts))
      counts = noneValue;
    rewriter.replaceOp(op, {outputY, indices, inverseIndices, counts});
    return success();
  }
};

void populateLoweringONNXUniqueOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXUniqueOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
