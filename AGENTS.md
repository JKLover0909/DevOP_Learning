# CLAUDE.md

## Tổng quan
DevOP_Learning là repo học DevOps/Docker kết hợp một script nghiên cứu: các ví dụ Docker (đang là con trỏ submodule rỗng) và `PPO2.py` — huấn luyện **PPO** (học tăng cường, PyTorch + Gymnasium) cho bài toán **đặt macro chip (placement)**, có vẻ phát triển từ dự án MaskPlace.

## Cấu trúc
- `PPO2.py` — mã huấn luyện/kiểm thử PPO: mạng CNN/ResNet18 làm encoder (`MyCNN`, `MyCNNCoarse`), `Actor`/`Critic`, lớp `PPO`, lưu placement (`save_placement`), nhiều tham số dòng lệnh (`--benchmark adaptec1`, `--pnm 128`, `--ordering_method`, `--centrality_file`, `--keep_top_n`, `--is_test`, `--model_path`...). Có ghi TensorBoard và đọc reward từ tên file `net_dict-...--{reward}.pkl`.
- `bindmount-apps/`, `multi-container-app/`, `python-docker-example/`, `welcome-to-docker/` — **gitlink (mode 160000) rỗng**, không có `.gitmodules` nên không clone được nội dung; thực chất là tham chiếu tới các repo mẫu Docker bên ngoài.

## Phụ thuộc không nằm trong repo
`PPO2.py` import `place_env`, `place_db` (`PlaceDB`), `comp_res` — các module (và dữ liệu benchmark adaptec) của dự án MaskPlace; cần có sẵn trong `PYTHONPATH`. Ngoài ra: `torch torchvision gymnasium numpy matplotlib tqdm tensorboard`, GPU CUDA khuyến nghị (script đặt `CUDA_VISIBLE_DEVICES=0`).

## Quy tắc khi làm việc
- Không tự ý sửa thuật toán/siêu tham số trong `PPO2.py` khi chưa được yêu cầu.
- Không commit kết quả huấn luyện lớn (`*.pkl`, `*.pth`, `runs/` đã trong `.gitignore`).
- Muốn dùng các ví dụ Docker: thêm lại submodule đúng URL (`git submodule add <url> <thư mục>`) hoặc gỡ các gitlink rỗng — hỏi trước khi thay đổi.
