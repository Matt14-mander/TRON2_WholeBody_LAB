# v4 快速播放与录像

`play_locomotion_v4.py`复用实体负载准备和严格checkpoint检查。默认读取新步态模型（policy123/history360/command3）；旧无步态模型必须加`--gait legacy`。

- `--quick`：连续预览，默认30秒，不跑10案例×多回合。移动角色，每3秒在训练速度范围内重新随机采样命令；跌倒或20秒超时由环境自动复位，录像继续。
- `--seconds 30 --command_interval 3`：设置总时长及换指令周期。
- `--base_command 0.15 0 0`：固定前进指令，自动启用quick。物理单位m/s、m/s、rad/s，不能超出训练范围。
- `--video`：自动启用quick及相机，输出 `preview.mp4`，叠加仿真时间、指令/实际速度、倾角和复位次数。
- `--fast`：以机器允许的速度执行，视频仍按50fps播放。启动仿真和RTX相机预热需要额外时间；输出视频时长由仿真时长决定。
- 不加 `--headless` 可在远程桌面显示窗口；纯SSH录像加 `--headless`。
- 默认确定性推理，不人为驱动机器人。随机命令不保证策略会走；静止且时间/指令叠字变化，可以明确区分策略停留与冻结的视频。

```bash
LIVESTREAM=0 python -u scripts/rsl_rl/play_locomotion_v4.py \
  --checkpoint_path /absolute/path/model_20000.pt \
  --gait legacy \
  --quick --video --fast --headless --seconds 30 --command_interval 3 \
  --output_dir logs/rsl_rl/v4_quick_$(date +%Y%m%d_%H%M%S) --device cuda:0
```

同目录另存 `preview_steps.jsonl`（逐控制步）及 `preview.json`（完成/帧数/复位次数）。quick模式不是正式验收，没有all_passed；不影响原评估标准。视频编码依赖 `imageio[ffmpeg]`、Pillow；出现缺包可在训练环境安装 `python -m pip install 'imageio[ffmpeg]' pillow`。相机先执行40次纯渲染预热，仍黑屏则报错，不静默保存黑视频。图形渲染慢时实际耗时可能超过视频时长。
