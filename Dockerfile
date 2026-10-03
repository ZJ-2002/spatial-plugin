# 共享 Python 工具镜像：hashutil / geo / pseudobulk / spatial / cmap / netprox 六个
# 插件家族共用。digest 由构建后 podman inspect 取出并写回各 manifest。
# 全部依赖均有 manylinux 轮子，无需编译工具链。
FROM docker.io/library/python:3.11-slim

RUN pip install --no-cache-dir \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        h5py \
        pyarrow \
        anndata \
        pyimzml \
        networkx \
        requests \
        blake3

# 中文注释：运行时节点全部 read_only_rootfs，无任何运行期安装动作。
