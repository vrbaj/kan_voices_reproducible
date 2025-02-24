import pickle
from pathlib import Path
import matplotlib.pyplot as plt

kan_women_arch = '126_51_2'
kan_men_arch = '115_184_138_2'
kan_women_settings = 'results_2layer_lamb0.001_g8_k3_100epochs'
kan_men_settings =  'results_2layer_lamb0.001_g5_k3_100epochs'

# Load pickled file
loss_path = Path('results_kan', kan_women_settings,
                 'training_data', 'women', kan_women_arch)  # Replace with your file path
loss_sum = 100 * [0]
plt.figure(figsize=(10, 6))
for idx, file_path in enumerate(loss_path.glob('*.pickle')):
    print(f'opening {file_path.name}')

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(data.keys())
    loss_values = data['train_loss']
    val_loss_values = data['test_loss']
    # Plot loss function values
    plt.plot(list(range(1,101)), val_loss_values, label=f'Loss cross-val split {idx + 1}')
    for idx, val_loss_value in enumerate(val_loss_values):
        loss_sum[idx] += val_loss_value
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Function Over Epochs')
plt.legend()
plt.grid()
plt.tight_layout()
plt.xticks([1] + list(range(10, 101, 10)))
plt.xlim([1, 100])
plt.savefig('kan_women_train.pdf', dpi=300, format='pdf')
plt.show()
plt.plot(loss_sum)
plt.show()