"""iALS с таргетом p=clip(mean(watch_ratio), 0, 1)**power вместо p=M."""
from .als import ImplicitALS


class CustomALS(ImplicitALS):
    def __init__(self, n_factors=32, reg=0.1, n_iter=15, alpha=10., power=1., random_state=42):
        """Задаёт общие параметры ALS и степень преобразования сигнала просмотра."""
        super().__init__(n_factors, reg, n_iter, alpha, random_state)
        if power <= 0:
            raise ValueError('power должен быть положительным')
        self.power = power

    def _targets(self):
        """Возвращает clip(mean(watch_ratio), 0, 1)**power для наблюдавшихся пар."""
        target = self.ratings_.copy()
        target.data = target.data.clip(0, 1)**self.power
        return target
