# Python IDE 的可复现 Pyright 分析镜像。
#
# Node 使用精确版本标签与已验证的官方镜像摘要；Python、NumPy 与 Pyright 的版本
# 也在本文件中固定，使开发机与后续部署环境分析同一份类型信息，而不是读取 Windows
# 主机的 Python 环境。
FROM node:22.22.1-bookworm-slim@sha256:4f77a690f2f8946ab16fe1e791a3ac0667ae1c3575c3e4d0d4589e9ed5bfaf3d AS lsp-runtime

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# 安装运行 Pyright 分析所需的 Linux Python，并在独立虚拟环境中固定 NumPy。
# Ruff 只读取 Gateway 通过 stdin 提供的当前文档并输出格式化文本；它不执行用户代码。
# apt 缓存会在同一层中删除，避免它成为镜像内容的一部分。
RUN apt-get update \
    && apt-get install --no-install-recommends --yes python3 python3-venv \
    && python3 -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir --disable-pip-version-check numpy==2.2.6 ruff==0.8.6 \
    && npm install --global --ignore-scripts pyright@1.1.411 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/server

# Gateway 的运行依赖与前端依赖隔离，避免把 ws 及其 lockfile 混入浏览器项目的 package.json。
COPY server/package.json server/package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts --no-audit --no-fund

COPY server/lspGateway.mjs ./lspGateway.mjs

# 容器服务使用固定 UID 的非 root 账户。/workspace 由 Compose 以只读方式挂载，
# 因此这里只为网关代码和运行时目录授予必要的读取与执行权限。
# 预先创建两个只读 bind mount 的目标路径。根文件系统运行时为只读，
# 因此 Docker 无法在容器启动阶段临时创建这些挂载点。
RUN groupadd --gid 10001 lsp \
    && useradd --uid 10001 --gid lsp --create-home --shell /usr/sbin/nologin lsp \
    && mkdir --parents /workspace/examples/lsp-workspace \
    && touch /workspace/pyrightconfig.json \
    && chown --recursive lsp:lsp /app /workspace

USER lsp:lsp
WORKDIR /workspace

# 默认命令是唯一允许启动的语言服务入口；Compose 只负责提供受限端口、只读挂载和资源边界。
CMD ["node", "/app/server/lspGateway.mjs"]
