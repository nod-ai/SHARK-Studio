#pragma once
#include <stddef.h>
#ifndef IREE_FILE_TOC
#define IREE_FILE_TOC

#if __cplusplus
extern "C" {
#endif // __cplusplus
typedef struct iree_file_toc_t {
  const char* name;             // the file's original name
  const char* data;             // beginning of the file
  size_t size;                  // length of the file
} iree_file_toc_t;
#if __cplusplus
}
#endif // __cplusplus

#endif  // IREE_FILE_TOC

#if __cplusplus
extern "C" {
#endif // __cplusplus
const iree_file_toc_t* simple_embedding_test_bytecode_module_cuda_c_create();
static inline size_t simple_embedding_test_bytecode_module_cuda_c_size() {
  return 1;
}
#if __cplusplus
}
#endif // __cplusplus

