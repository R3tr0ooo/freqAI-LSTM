# 策略修复总结

## 修复的问题

### 1. 100%资金使用问题
**原因**：
- config.json中设置了`"stake_amount": "unlimited"`，导致使用全部资金
- 当模型置信度为0时，计算逻辑仍可能返回接近100%的资金使用率

**修复**：
- 将config.json中的`stake_amount`改为固定值20 USDT
- 修改`calculate_adaptive_stake_ratio`方法，当置信度或目标值太低时返回None
- 在`custom_stake_amount`中，当stake_ratio为None时使用config设置的值

### 2. 1倍杠杆问题
**原因**：
- 当模型置信度为0.000时，杠杆计算返回最小值1倍
- 缺少合理的默认值处理

**修复**：
- 在`calculate_adaptive_leverage`中增加阈值检查
- 当目标值或置信度太低时，返回默认10倍杠杆
- 增加详细的调试日志

## 修复后的行为

### 杠杆计算逻辑
```python
1. 如果adaptive_leverage_enabled = False，使用global_leverage（默认10倍）
2. 如果数据不足（少于20条），返回10倍
3. 如果目标值 < 0.01 或 置信度 < 0.01，返回10倍
4. 否则，基于模型输出计算杠杆（1-100倍范围）
```

### 资金管理逻辑
```python
1. 如果adaptive_position_enabled = False，使用config中的stake_amount
2. 如果数据不足（少于20条），使用config设置
3. 如果目标值 < 0.01 或 置信度 < 0.01，使用config设置
4. 否则，基于模型输出计算资金比例（1%-100%范围）
```

## 日志改进

增加了更详细的调试日志：
- 杠杆计算时输出target_strength、confidence和最终杠杆值
- 资金计算时输出stake_ratio百分比
- 异常情况时明确说明使用的fallback值

## 配置建议

1. **固定资金模式**：
   - 设置`stake_amount: 20`（或其他固定值）
   - 设置`adaptive_position_enabled: false`

2. **自适应资金模式**：
   - 设置`stake_amount: 20`作为fallback值
   - 设置`adaptive_position_enabled: true`
   - 模型会根据置信度动态调整资金使用（1%-100%）

3. **杠杆设置**：
   - `adaptive_leverage_enabled: true`：自适应杠杆（1-100倍）
   - `adaptive_leverage_enabled: false`：固定10倍杠杆
