from Models.LLaMA_Adapter import LLaMA_adapter
from Utils.misc import MetricLogger

class EarlyStopper:
    """早停策略实现"""
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: 耐心轮次，没有改善后停止。
            min_delta: 最小改善阈值。
            mode: 'min' (最小化损失) 或 'max' (最大化准确率)。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
    
    def get_state(self):
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop
        }

    def load_state(self, state_dict):
        self.patience = state_dict['patience']
        self.min_delta = state_dict['min_delta']
        self.mode = state_dict['mode']
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']

    def __call__(self, model: LLaMA_adapter, var_logger: MetricLogger, epoch: int, data_iter_step: int) -> bool:
        """检查是否应该早停"""
        var_logger.synchronize_between_processes()
        val_loss = var_logger.meters['loss'].deque[-1]
        if self.mode == 'min':
            score = val_loss
        else:  # mode == 'max'
            score = -val_loss

        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            model.best_model_state_dict = (f'Epoch_{epoch}_{data_iter_step}', {name: param.data.copy() for name, param in model.named_parameters() if param.requires_grad})
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop