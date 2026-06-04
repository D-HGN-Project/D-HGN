"""
GPU加速配置和优化脚本
提供GPU检测、配置和性能优化功能
"""
import torch
import torch.backends.cudnn as cudnn


def setup_gpu(use_cpu=False, gpu_id=0):
    """
    配置GPU设备
    
    Args:
        use_cpu: 是否强制使用CPU
        gpu_id: GPU设备ID（如果有多个GPU）
        
    Returns:
        device: torch.device对象
    """
    if use_cpu:
        device = torch.device('cpu')
        print("🖥️  使用CPU进行训练")
    else:
        if torch.cuda.is_available():
            # 设置使用的GPU
            torch.cuda.set_device(gpu_id)
            device = torch.device(f'cuda:{gpu_id}')
            
            # 打印GPU信息
            print(f"🚀 使用GPU加速训练")
            print(f"   GPU设备: {torch.cuda.get_device_name(gpu_id)}")
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   PyTorch版本: {torch.__version__}")
            print(f"   可用GPU数量: {torch.cuda.device_count()}")
            print(f"   当前GPU内存: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f} GB")
            
            # 启用cudnn加速
            cudnn.benchmark = True  # 自动寻找最优卷积算法
            cudnn.enabled = True
            print("   ✓ cuDNN加速已启用")
            
        else:
            device = torch.device('cpu')
            print("⚠️  CUDA不可用，使用CPU进行训练")
            print("   提示: 安装GPU版本的PyTorch以启用GPU加速")
            print("   访问: https://pytorch.org/get-started/locally/")
    
    return device


def check_gpu_memory():
    """检查GPU内存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  总内存: {props.total_memory / 1024**3:.2f} GB")
            print(f"  已分配: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  已缓存: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  可用内存: {(props.total_memory - torch.cuda.memory_allocated(i)) / 1024**3:.2f} GB")


def optimize_gpu_settings():
    """优化GPU设置以提高性能"""
    if torch.cuda.is_available():
        # 启用TF32（适用于Ampere架构及以上）
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
            print("✓ TF32加速已启用（适用于RTX 30系列及以上）")
        
        # 启用cudnn自动调优
        torch.backends.cudnn.benchmark = True
        
        # 设置cudnn确定性（如果需要可复现性，设为True会稍慢）
        torch.backends.cudnn.deterministic = False
        
        print("✓ GPU优化设置已完成")


def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ GPU缓存已清理")


def get_optimal_batch_size(model, input_shape, device, max_batch_size=128):
    """
    自动测试最优batch size
    
    Args:
        model: 模型
        input_shape: 输入形状 (不含batch维度)
        device: 设备
        max_batch_size: 最大测试batch size
        
    Returns:
        optimal_batch_size: 最优batch size
    """
    if device.type == 'cpu':
        return 8  # CPU默认使用较小batch size
    
    model.eval()
    batch_size = 1
    
    print("🔍 正在测试最优batch size...")
    
    while batch_size <= max_batch_size:
        try:
            # 创建测试数据
            test_input = torch.randn(batch_size, *input_shape).to(device)
            
            # 前向传播测试
            with torch.no_grad():
                _ = model(test_input)
            
            # 清理缓存
            del test_input
            torch.cuda.empty_cache()
            
            # 检查内存使用
            memory_used = torch.cuda.memory_allocated(device) / 1024**3
            memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            
            if memory_used / memory_total > 0.8:  # 如果使用超过80%内存，停止
                break
            
            print(f"   Batch size {batch_size}: ✓ (内存使用: {memory_used:.2f}/{memory_total:.2f} GB)")
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   Batch size {batch_size}: ✗ (内存不足)")
                break
            else:
                raise e
    
    optimal_batch_size = max(1, batch_size // 2)
    print(f"✓ 推荐batch size: {optimal_batch_size}")
    
    return optimal_batch_size


def enable_mixed_precision():
    """
    启用混合精度训练（FP16）
    可以显著加速训练并减少内存使用
    
    Returns:
        scaler: GradScaler对象
    """
    if torch.cuda.is_available():
        # 兼容新旧版本PyTorch
        if hasattr(torch.amp, 'GradScaler'):
            scaler = torch.amp.GradScaler('cuda')
        else:
            scaler = torch.cuda.amp.GradScaler()
            
        print("✓ 混合精度训练(FP16)已启用")
        print("   预期加速: 1.5-2x")
        print("   内存节省: ~30-50%")
        return scaler
    else:
        return None


class GPUMonitor:
    """GPU监控器"""
    
    def __init__(self, device):
        self.device = device
        self.enabled = device.type == 'cuda'
    
    def log_memory(self, step_name=""):
        """记录当前GPU内存使用"""
        if self.enabled:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            print(f"[{step_name}] GPU内存 - 已分配: {allocated:.2f}GB, 已缓存: {reserved:.2f}GB")
    
    def reset_peak_memory(self):
        """重置峰值内存统计"""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def get_peak_memory(self):
        """获取峰值内存使用"""
        if self.enabled:
            peak = torch.cuda.max_memory_allocated(self.device) / 1024**3
            return peak
        return 0


# 使用示例
if __name__ == "__main__":
    print("="*50)
    print("GPU配置和优化工具")
    print("="*50)
    
    # 1. 设置GPU
    device = setup_gpu(use_cpu=False, gpu_id=0)
    
    # 2. 检查GPU内存
    check_gpu_memory()
    
    # 3. 优化GPU设置
    optimize_gpu_settings()
    
    # 4. 启用混合精度
    scaler = enable_mixed_precision()
    
    # 5. 创建GPU监控器
    monitor = GPUMonitor(device)
    
    print("\n✅ GPU配置完成！")
    print("\n使用建议:")
    print("1. 在train_dhgn.py中添加 --gpu_id 参数指定GPU")
    print("2. 使用混合精度训练可以加速1.5-2倍")
    print("3. 根据GPU内存调整batch_size")
    print("4. 监控GPU内存使用避免OOM错误")
