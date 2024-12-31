## 均线交叉策略通知
基于 Freqtrade 框架的均线交叉策略实时通知系统。

## 功能特点
- 实时监控多个时间周期的均线交叉
- 支持多币种监控
- 自动发送微信群通知
- 清晰的趋势指示（上升↗️、下降↘️、横盘➖）
- 多时间周期分析（1m, 5m, 15m, 1h, 4h, 1d）
## 通知格式
- 币种与时间周期标识
- 各周期趋势指示
- 均线交叉信号
- 实时价格信息
## 技术框架
- 基于 Freqtrade 交易框架[Freqtrade](https://github.com/freqtrade/freqtrade)
- 微信群通知集成 [wechatferry]([(https://github.com/wechatferry/wechatferry)]
- Python 实现
## 部署要求
- Python 3.10+
- Freqtrade 环境
- 微信通知配置
# 贡献
- 欢迎提交 Pull Requests 或 Issues 来完善此策略！
- 项目地址：[strategy]([((https://github.com/KGLongWang/freqtrade))]

# 许可证
- MIT License

# 自己部署的wx通知群
<img src="https://github.com/user-attachments/assets/48894d72-c3a7-49c8-abdc-e3d90bb08044" width="200"/>

注：此项目仅供学习研究使用，不构成投资建议。请谨慎使用，风险自负。
