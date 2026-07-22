// In-process CoreML runtime shim (BERT Phase 4, ANE variant) — macOS only.
//
// BNNSGraph executes mlmodelc on CPU by design; Apple gates ANE dispatch
// behind the CoreML runtime. This shim exposes the three C entry points the
// Rust engine needs to run the SAME compiled artifacts on the Neural Engine:
//
//   nue_coreml_load(path, units)  -> retained MLModel* (NULL on failure)
//   nue_coreml_predict(...)       -> 0 ok / negative stage code
//   nue_coreml_free(handle)
//
// Inputs are wrapped as external-data MLMultiArrays (zero-copy over the
// caller's f32 buffers, strides in ELEMENTS); the output is copied back into
// the caller's buffer. Feature names match the authored mlprogram: h, bias,
// out. Compiled by build.rs via cc with -fobjc-arc; CoreML + Foundation are
// linked as frameworks there.

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

void *nue_coreml_load(const char *path, int compute_units) {
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        switch (compute_units) {
            case 1: cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine; break;
            case 2: cfg.computeUnits = MLComputeUnitsAll; break;
            default: cfg.computeUnits = MLComputeUnitsCPUOnly; break;
        }
        NSError *err = nil;
        MLModel *m = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
        if (!m) {
            if (getenv("RZT_DBG_GRAPH")) NSLog(@"[coreml] load failed: %@", err);
            return NULL;
        }
        return (__bridge_retained void *)m;
    }
}

int nue_coreml_predict(void *handle, const float *h, const float *bias, float *out, long s,
                       long hidden) {
    @autoreleasepool {
        MLModel *m = (__bridge MLModel *)handle;
        NSError *err = nil;
        MLMultiArray *ha = [[MLMultiArray alloc] initWithDataPointer:(void *)h
                                                               shape:@[ @(s), @(hidden) ]
                                                            dataType:MLMultiArrayDataTypeFloat32
                                                             strides:@[ @(hidden), @1 ]
                                                         deallocator:nil
                                                               error:&err];
        if (!ha) return -3;
        MLMultiArray *ba = [[MLMultiArray alloc] initWithDataPointer:(void *)bias
                                                               shape:@[ @(s) ]
                                                            dataType:MLMultiArrayDataTypeFloat32
                                                             strides:@[ @1 ]
                                                         deallocator:nil
                                                               error:&err];
        if (!ba) return -4;
        MLDictionaryFeatureProvider *in = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{
                @"h" : [MLFeatureValue featureValueWithMultiArray:ha],
                @"bias" : [MLFeatureValue featureValueWithMultiArray:ba]
            }
                         error:&err];
        if (!in) return -5;
        id<MLFeatureProvider> res = [m predictionFromFeatures:in error:&err];
        if (!res) {
            if (getenv("RZT_DBG_GRAPH")) NSLog(@"[coreml] predict failed: %@", err);
            return -6;
        }
        MLMultiArray *oa = [res featureValueForName:@"out"].multiArrayValue;
        if (!oa) return -7;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        memcpy(out, oa.dataPointer, (size_t)(s * hidden) * sizeof(float));
#pragma clang diagnostic pop
        return 0;
    }
}

void nue_coreml_free(void *handle) {
    if (handle) {
        MLModel *m = (__bridge_transfer MLModel *)handle;
        (void)m;
    }
}

// Llama prefill: one input h [s, hidden]; outputs "out" [s, hidden] plus
// per-layer "k{i}f"/"v{i}f" [s, kv_heads, head_dim] (K post-rope, cache
// layout). The caller provides ONE contiguous kv block laid out
// [layers][2][s*kv_heads*head_dim] f32 — k at slot 2i, v at 2i+1.
int nue_coreml_predict_prefill(void *handle, const float *h, float *out, float *kv, long s,
                               long hidden, long layers, long kv_heads, long head_dim) {
    @autoreleasepool {
        MLModel *m = (__bridge MLModel *)handle;
        NSError *err = nil;
        MLMultiArray *ha = [[MLMultiArray alloc] initWithDataPointer:(void *)h
                                                               shape:@[ @(s), @(hidden) ]
                                                            dataType:MLMultiArrayDataTypeFloat32
                                                             strides:@[ @(hidden), @1 ]
                                                         deallocator:nil
                                                               error:&err];
        if (!ha) return -3;
        MLDictionaryFeatureProvider *in = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{@"h" : [MLFeatureValue featureValueWithMultiArray:ha]}
                         error:&err];
        if (!in) return -5;
        id<MLFeatureProvider> res = [m predictionFromFeatures:in error:&err];
        if (!res) {
            if (getenv("RZT_DBG_GRAPH")) NSLog(@"[coreml] prefill predict failed: %@", err);
            return -6;
        }
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        MLMultiArray *oa = [res featureValueForName:@"out"].multiArrayValue;
        if (!oa) return -7;
        memcpy(out, oa.dataPointer, (size_t)(s * hidden) * sizeof(float));
        size_t kvsz = (size_t)(s * kv_heads * head_dim);
        for (long i = 0; i < layers; i++) {
            NSString *kn = [NSString stringWithFormat:@"k%ldf", i];
            NSString *vn = [NSString stringWithFormat:@"v%ldf", i];
            MLMultiArray *ka = [res featureValueForName:kn].multiArrayValue;
            MLMultiArray *va = [res featureValueForName:vn].multiArrayValue;
            if (!ka || !va) return -8;
            memcpy(kv + (size_t)(2 * i) * kvsz, ka.dataPointer, kvsz * sizeof(float));
            memcpy(kv + (size_t)(2 * i + 1) * kvsz, va.dataPointer, kvsz * sizeof(float));
        }
#pragma clang diagnostic pop
        return 0;
    }
}
