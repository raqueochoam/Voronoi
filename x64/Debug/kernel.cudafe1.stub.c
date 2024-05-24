#define __NV_MODULE_ID _af638eb6_9_kernel_cu_ec75798e
#define __NV_CUBIN_HANDLE_STORAGE__ extern
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "kernel.fatbin.c"
extern void __device_stub__Z13voronoiKernelP5PointiPh(struct Point *, int, unsigned char *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void);
#pragma section(".CRT$XCT",read)
__declspec(allocate(".CRT$XCT"))static void (*__dummy_static_init__sti____cudaRegisterAll[])(void) = {__sti____cudaRegisterAll};
void __device_stub__Z13voronoiKernelP5PointiPh(
struct Point *__par0, 
int __par1, 
unsigned char *__par2)
{
__cudaLaunchPrologue(3);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaLaunch(((char *)((void ( *)(struct Point *, int, unsigned char *))voronoiKernel)));
}
void voronoiKernel( struct Point *__cuda_0,int __cuda_1,unsigned char *__cuda_2)
{__device_stub__Z13voronoiKernelP5PointiPh( __cuda_0,__cuda_1,__cuda_2);
}
#line 1 "x64/Debug/kernel.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback(
void **__T3)
{
__nv_dummy_param_ref(__T3);
__nv_save_fatbinhandle_for_managed_rt(__T3);
__cudaRegisterEntry(__T3, ((void ( *)(struct Point *, int, unsigned char *))voronoiKernel), _Z13voronoiKernelP5PointiPh, (-1));
}
static void __sti____cudaRegisterAll(void)
{
____cudaRegisterLinkedBinary(__nv_cudaEntityRegisterCallback);
}
