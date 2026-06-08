# Photo Search

基于 SigLIP2 的本地图片搜索系统。

## 安装

```bash
cd photo-search
uv sync
```

如果你仍然使用 `pip`，需要自己处理 `torch` 和 `onnxruntime` 的平台差异。

## 使用

### 1. 编辑配置

编辑 `config.yaml`，把你的图片目录加进去：

```yaml
photo_dirs:
  - "C:/Users/你/Pictures"
  - "D:/照片"
```

模型路径也在这里配置（支持 HuggingFace id 或本地路径）。

### 2. 索引

```bash
# 索引配置文件中的所有目录
python run_indexer.py --scan-all

# 索引指定目录
python run_indexer.py ~/Photos

# 监控模式（自动索引新文件）
python run_indexer.py --watch ~/Photos
```

### 3. 启动搜索服务

```bash
python run_server.py
```

浏览器打开 `http://localhost:8080`。

### 搜索流程

1. 输入文本关键词，如“鸟”“猫”“日落”
2. 手动填写时间 / 地点 / 设备条件
3. SigLIP2 做向量搜索
4. LanceDB 同时做向量搜索 + 元数据过滤

## 项目结构

```
photo-search/
├── config.yaml           # 配置（目录、模型、端口）
├── run_indexer.py        # 索引服务入口
├── run_server.py         # Web 服务入口
├── shared/
│   ├── config.py         # 配置加载
│   └── store.py          # LanceDB 存储
├── indexer/
│   ├── exif.py           # EXIF 提取
│   ├── encoder.py        # SigLIP2 编码
│   ├── geocoder.py       # GPS 反向编码
│   └── indexer.py        # 索引逻辑
├── api/
│   ├── app.py            # FastAPI 应用
│   ├── search.py         # 搜索逻辑
│   └── models.py         # 数据模型
└── web/
    ├── index.html
    ├── style.css
    └── app.js
```
