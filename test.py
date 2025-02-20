import torch

def main():
    # 建立兩個 tensor
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])
    
    # 進行加法運算
    z = x + y
    print("x:", x)
    print("y:", y)
    print("x + y:", z)
    
    # 如果有 GPU 可用，則在 GPU 上進行運算
    if torch.cuda.is_available():
        x_gpu = x.to("cuda")
        y_gpu = y.to("cuda")
        z_gpu = x_gpu + y_gpu
        print("在 GPU 上 x + y:", z_gpu)
    
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    print(torch.version.cuda)

if __name__ == "__main__":
    main()
