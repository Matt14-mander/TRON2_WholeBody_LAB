# SDK estimator 采样调度修复（2026-10-12）

服务器回报：sample_time_gap=129，周期 P50=20.001 ms、P95=22.880 ms、max=25.650 ms；状态年龄最大 0.985 ms。低有效率主要由间隔超出 18–22 ms 后四帧历史反复清空造成，不是旧状态问题。同步推理/JSONL flush 是可能的调度负担，但服务器尚未完成对照，不能认定为唯一原因。

修复文件：deploy 的 controllers/SFYGController.py、controllers/sfyg_intent.py。

estimator 改为有界 FIFO 后台线程；采样按单调时钟固定相位的 20 ms deadline，与 policy decimation 分开。采样时复制状态、前一发送目标、前一 OCS2 solution，时间戳仍是采样时刻；队列积压最多 2 帧，满时丢弃新帧并计数，后续时间间隙仍会打断历史。保持 18–22 ms 有效间隔检查，不插值、不复制旧状态，不改变 policy、future wrench、负载或 PD 参数。停机先发布阻尼，再结束 observer。

本地 23 项 deploy 测试通过，包括异步状态快照、队列溢出计数、固定相位与跳过漏帧。只验证实现行为，未实测服务器有效帧率或带负载稳定性。

服务器停掉控制器后先备份：

```bash
cd "$HOME/workspace/limx_deploy/tron2_rl_deploy_python"
BACKUP_DIR="$PWD/backup_intent_timing_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp controllers/SFYGController.py controllers/sfyg_intent.py "$BACKUP_DIR/"
```

Windows PowerShell 发送本地新版两文件：

```powershell
scp 'E:\Robot\Sustech-多人形协作\tron2_rl_deploy_python\controllers\SFYGController.py' 'E:\Robot\Sustech-多人形协作\tron2_rl_deploy_python\controllers\sfyg_intent.py' matt14@server-Super-Server:~/workspace/limx_deploy/tron2_rl_deploy_python/controllers/
```

重启带负载仿真，再运行此前同一条 OCS2/estimator 命令，并使用新 INTENT_LOG。日志末尾 observer_summary 包含 queue_dropped_samples；控制台打印 worker stopped、error。复测有效率、间隔分布及 reset 原因；若仍大量越界，继续定位主控制线程/ONNX 的耗时，不通过放宽阈值伪造改善。估计零力偏置是另一个问题，本次调度修复不解决精度迁移。
