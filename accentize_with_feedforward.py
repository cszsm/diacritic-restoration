from src.accentizer import accentize_with_feedforward
from os.path import join

accentized = accentize_with_feedforward('arvizturo tukorfurogep',
                                        join('100', '10', '0'))
print(accentized)