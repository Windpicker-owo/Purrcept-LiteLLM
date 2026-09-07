# 发布 Purrcept LiteLLM

发布 GitHub Release 后，`.github/workflows/publish.yml` 自动构建并发布到 PyPI。
仅推送代码或 tag 不会发布。Release tag 必须与 `pyproject.toml` 中的版本完全对应，
格式为 `v版本号`；本次为 `v0.1.3`。

## 首次配置

在 [PyPI 项目发布设置](https://pypi.org/manage/project/purrcept-litellm/settings/publishing/)
添加 GitHub Trusted Publisher：

| 字段 | 值 |
| --- | --- |
| Owner | `Windpicker-owo` |
| Repository | `Purrcept-LiteLLM` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

GitHub 仓库须有 `pypi` Environment，无需保存长期 PyPI token。
认证方式见 [PyPI 官方说明](https://docs.pypi.org/trusted-publishers/using-a-publisher/)。

## 发布步骤

1. 更新包版本、锁文件、README 和 CHANGELOG，提交并推送。
2. 确保声明的 Core 版本已在 PyPI 发布。本次至少需要 `purrcept_core 0.5.1`。
3. 在该提交创建对应 tag 并发布 GitHub Release。
4. 工作流校验版本，使用 `uv sync --no-sources` 安装公开依赖，运行 Ruff、Pyright 和
   离线测试，执行现有 100% 覆盖率门禁，然后构建 sdist 及由 sdist 构建的 wheel。
   `--no-sources` 防止 CI 错把开发机上的相邻 Core 仓库当作已发布依赖。
5. Twine 检查通过后，独立的 `pypi` Job 下载构建产物，以 OIDC 身份发布。
   该 Job 不检出或执行项目源码，构建 Job 不持有发布身份。
6. 确认工作流成功，并检查 PyPI 上的版本与两个发行文件。

不移动已发布的 tag。代码或包内容需要修正时使用新版本；若仅为认证配置或临时服务问题，
确认 PyPI 上传状态后重跑失败 Job。PyPI 已发布的文件不能覆盖。
GitHub prerelease 也触发此工作流，需要预发布包时应在包元数据中使用 PEP 440 预发布版本。
