# HGG-TopK-Training 新手入门指南

本指南面向第一次使用本项目的同学，按步骤操作即可在 10 分钟内跑通基础实验，并能根据模型/算法/稀疏率快速切换。

一、环境准备
- 推荐环境：Linux/WSL + NVIDIA GPU（分布式后端固定 
ccl，Windows 原生通常不支持；单卡可用 --gpus 1）
- Python 3.8+；安装依赖：
  pip install -r requirements.txt
- 数据：
  - CIFAR10/100：由 torchvision 自动下载到 ./data
  - PTB（LSTM）：准备 ptb.train.txt/ptb.valid.txt/ptb.test.txt 放到 --data-dir（默认 ./data）
  - GPT‑2（WikiText‑2/OpenWebText）：首次需从 HuggingFace 下载（需网络）；离线请预放缓存

二、最快开始（0–10 分钟）
- 交互菜单（推荐）：一键跑常见实验/可视化
  python run.py
- 3 方法快速对比（默认 5 个 epoch）
  python quick_compare.py --epochs 5 --gpus 1
- GPT‑2 小测（baseline/TopK/HGG‑TopK）
  python test_gpt2.py

三、单次实验模板（最常用）
- 视觉（CIFAR‑10/100）
  1) Baseline（不压缩）
     python trainers/trainer.py --model resnet18 --dataset cifar10 --epochs 50 --gpus 1
  2) TopK（5%）
     python trainers/trainer.py --model resnet18 --dataset cifar10 --compressor topk --density 0.05 --epochs 50 --gpus 1
  3) HGG‑TopK（5%，推荐）
     python trainers/trainer.py --model resnet18 --dataset cifar10 --compressor hggtopk --density 0.05 --epochs 50 --gpus 1
  4) HGG‑TopK + 流水线（仅 hggtopk 支持）
     python trainers/trainer.py --model resnet18 --dataset cifar10 --compressor hggtopk --density 0.05 --use-pipeline --epochs 50 --gpus 1
- 语言
  1) GPT‑2 Small + WikiText‑2（baseline）
     python trainers/trainer.py --model gpt2-small --dataset wikitext2 --batch-size 4 --epochs 2 --log-interval 50 --gpus 1
  2) GPT‑2 Small + HGG‑TopK（5%）
     python trainers/trainer.py --model gpt2-small --dataset wikitext2 --compressor hggtopk --density 0.05 --batch-size 4 --epochs 2 --log-interval 50 --gpus 1
  3) LSTM（PTB）
     python trainers/trainer.py --model lstm --dataset ptb --batch-size 32 --epochs 40 --gpus 1 --data-dir ./data

四、批量对比与论文套件
- 快速自定义对比：
  python quick_compare.py --model resnet18 --dataset cifar10 --epochs 5 --density 0.05 --methods baseline topk gaussian hggtopk --gpus 1
- 论文级整套实验（交互选择 1–8；9 为全部）
  python experiments/run_experiments.py

五、结果与可视化
- 结果 JSON：./logs/{model}_{dataset}_{compressor}_d{density}[_pipeline].json
- 可视化/报告：
  - 概览：python visualization/visualizer.py --summary
  - 对比通信/稀疏化耗时：python visualization/visualizer.py --compare-comm --compare-sparse
  - 全部 PNG 图：python visualization/visualizer.py --plot --output-dir ./figures
  - 文本报告：python visualization/visualizer.py --report

六、常用参数（trainers/trainer.py）
- 模型：--model resnet18|resnet50|vgg11|vgg16|mobilenet|lstm|gpt2-small|gpt2-medium
- 数据：--dataset cifar10|cifar100|ptb|wikitext2|openwebtext
- 压缩算法：--compressor topk|gaussian|dgcsampling|redsync|hggtopk|randomk|topk2|gaussian2
- 稀疏率：--density 0–1（如 0.05 = 5%）
- 流水线：--use-pipeline（仅 hggtopk）
- 训练：--epochs、--batch-size、--lr；多卡：--gpus N；目录：--data-dir、--log-dir

七、推荐配方（复制即可用）
- 视觉三法对比（5 epoch）：
  python quick_compare.py --model resnet18 --dataset cifar10 --epochs 5 --gpus 1
- HGG‑TopK 主实验（5%）：
  python trainers/trainer.py --model resnet18 --dataset cifar10 --compressor hggtopk --density 0.05 --epochs 50 --gpus 1
- GPT‑2 小测（5%）：
  python trainers/trainer.py --model gpt2-small --dataset wikitext2 --compressor hggtopk --density 0.05 --batch-size 4 --epochs 2 --log-interval 50 --gpus 1

八、常见问题
- NCCL/分布式初始化失败：在 Linux/WSL 跑；单卡先用 --gpus 1
- CUDA OOM：降低 --batch-size（如 128→64→32），或改小模型（resnet18）
- GPT‑2 下载慢：需要网络；离线预放 HF 缓存
- 可视化无结果：确认 ./logs 下有对应 JSON，调用时加 --log-dir 指向该目录

九、目录速览
- 核心训练：trainers/trainer.py（统一入口）
- 压缩算法：core/compression.py（含 HGG‑TopK 优化实现）
- 流水线：core/hgg_pipeline.py
- 语言模型/数据：core/models.py、data_utils/gpt2_data.py、data_utils/ptb_reader.py
- 快速对比/论文套件：quick_compare.py、experiments/*.py
- 可视化：visualization/visualizer.py

建议：第一次使用先跑 python run.py 或上面的“推荐配方”，再逐步替换模型/算法/稀疏率，观察 ./logs 与图表变化。
