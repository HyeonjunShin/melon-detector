import matplotlib.pyplot as plt
import torch
from torchvision.io import ImageReadMode, read_image


def visualize_sequence(batch_imgs, batch_idx=0):
    # batch_imgs: [B, T, C, H, W]
    seq_img = batch_imgs.cpu()  # 해당 배치 선택
    T, C, H, W = seq_img.shape

    # 2행 5열 설정
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(T):
        # [C, H, W] -> [H, W, C]로 변환 및 정규화 해제(필요시)
        img = seq_img[i].permute(1, 2, 0).numpy()

        # 만약 이미지가 0~1 사이로 정규화되어 있지 않다면 clip
        # img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(f"Frame {i}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    get_color_path = lambda x: (
        f"/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_train_dataset/rgb_{x:05d}.png"
    )
    get_segmentation_path = lambda x: (
        f"/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_train_dataset/instance_segmentation_{x:05d}.png"
    )
    # color = read_image(get_color_path(0), mode=ImageReadMode.RGB)

    random_idx = 0
    while True:
        s = random_idx * 10
        e = (random_idx + 1) * 10
        print(s, e)
        colors = [read_image(get_color_path(i), mode=ImageReadMode.RGB) for i in range(s, e)]
        colors = torch.stack(colors, dim=0)
        print(colors.shape)
        visualize_sequence(colors)
        random_idx += 10 


if __name__ == "__main__":
    main()
