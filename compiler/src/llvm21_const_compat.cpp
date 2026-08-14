#include "llvm-c/Core.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CBindingWrapping.h"

using namespace llvm;

extern "C" LLVMValueRef LLVMConstMul(LLVMValueRef LHSConstant,
                                      LLVMValueRef RHSConstant) {
  auto *LHS = cast<Constant>(unwrap(LHSConstant));
  auto *RHS = cast<Constant>(unwrap(RHSConstant));
  return wrap(ConstantExpr::get(Instruction::Mul, LHS, RHS));
}

extern "C" LLVMValueRef LLVMConstNSWMul(LLVMValueRef LHSConstant,
                                         LLVMValueRef RHSConstant) {
  auto *LHS = cast<Constant>(unwrap(LHSConstant));
  auto *RHS = cast<Constant>(unwrap(RHSConstant));
  return wrap(ConstantExpr::get(Instruction::Mul, LHS, RHS,
                                OverflowingBinaryOperator::NoSignedWrap));
}

extern "C" LLVMValueRef LLVMConstNUWMul(LLVMValueRef LHSConstant,
                                         LLVMValueRef RHSConstant) {
  auto *LHS = cast<Constant>(unwrap(LHSConstant));
  auto *RHS = cast<Constant>(unwrap(RHSConstant));
  return wrap(ConstantExpr::get(Instruction::Mul, LHS, RHS,
                                OverflowingBinaryOperator::NoUnsignedWrap));
}
