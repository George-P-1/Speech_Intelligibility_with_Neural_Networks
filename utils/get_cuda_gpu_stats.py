import torch

def print_cuda_gpu_stats():
    if torch.cuda.is_available():
        print("CUDA is available.")
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        
        for device in range(device_count):
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA is not available.")

# Call the function to print the CUDA GPU stats
print_cuda_gpu_stats()