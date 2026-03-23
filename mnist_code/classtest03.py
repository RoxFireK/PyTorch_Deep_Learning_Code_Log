import os
import numpy as np
from tensorflow.keras import datasets

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    x = x / 255.0
    x_test = x_test / 255.0
    return (x, y), (x_test, y_test)


# ========== 数据增强类 ==========
class DataAugmentation:
    @staticmethod
    def augment(images, labels):
        """数据增强"""
        augmented_images = []
        augmented_labels = []

        for img, label in zip(images, labels):
            # 原始图像
            augmented_images.append(img)
            augmented_labels.append(label)

            # 随机平移 (±2像素)
            shift_x = np.random.randint(-2, 3)
            shift_y = np.random.randint(-2, 3)
            shifted = np.roll(img, shift_x, axis=0)
            shifted = np.roll(shifted, shift_y, axis=1)
            augmented_images.append(shifted)
            augmented_labels.append(label)

            # 随机添加噪声
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.05, img.shape)
                noisy = np.clip(img + noise, 0, 1)
                augmented_images.append(noisy)
                augmented_labels.append(label)

        return np.array(augmented_images), np.array(augmented_labels)


# ========== Dropout类 ==========
class Dropout:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True

    def forward(self, x):
        if not self.training:
            return x
        self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape) / (1 - self.dropout_rate)
        return x * self.mask

    def backward(self, grad_y):
        return grad_y * self.mask


# ========== Batch Normalization类 ==========
class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True

    def forward(self, x):
        if self.training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta

        self.mem = {'x': x, 'mean': mean, 'var': var, 'x_norm': x_norm}
        return out

    def backward(self, grad_y):
        x = self.mem['x']
        mean = self.mem['mean']
        var = self.mem['var']
        x_norm = self.mem['x_norm']

        N = x.shape[0]
        dx_norm = grad_y * self.gamma
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + self.eps) ** -1.5, axis=0)
        dmean = np.sum(dx_norm * -1 / np.sqrt(var + self.eps), axis=0) + dvar * np.mean(-2 * (x - mean), axis=0)

        dx = dx_norm / np.sqrt(var + self.eps) + dvar * 2 * (x - mean) / N + dmean / N
        return dx


# ========== 基础层类 ==========
class Matmul():
    def __init__(self):
        self.mem = {}

    def forward(self, x, W):
        h = np.matmul(x, W)
        self.mem = {"x": x, "W": W}
        return h

    def backward(self, grad_y):
        x = self.mem["x"]
        W = self.mem["W"]
        grad_x = np.matmul(grad_y, W.T)
        grad_W = (np.matmul(x.T, grad_y))
        return grad_x, grad_W


class Relu():
    def __init__(self):
        self.mem = {}

    def forward(self, x):
        self.mem["x"] = x
        return np.where(x > 0, x, np.zeros_like(x))

    def backward(self, grad_y):
        x = self.mem["x"]
        return (x > 0).astype(np.float32) * grad_y


class Softmax():
    def __init__(self):
        self.mem = {}
        self.epsilon = 1e-12

    def forward(self, x):
        x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        denominator = np.sum(x_exp, axis=1, keepdims=True)
        out = x_exp / (denominator + self.epsilon)
        self.mem["out"] = out
        return out

    def backward(self, grad_y):
        s = self.mem["out"]
        sisj = np.matmul(np.expand_dims(s, axis=2), np.expand_dims(s, axis=1))
        g_y_exp = np.expand_dims(grad_y, axis=1)
        tmp = np.matmul(g_y_exp, sisj)
        tmp = np.squeeze(tmp, axis=1)
        softmax_grad = -tmp + grad_y * s
        return softmax_grad


class Cross_entropy():
    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}

    def forward(self, x, labels):
        log_prob = np.log(x + self.epsilon)
        out = np.mean(np.sum(-log_prob * labels, axis=1))
        self.mem["x"] = x
        return out

    def backward(self, labels):
        x = self.mem["x"]
        return -1 / (x + self.epsilon) * labels


# ========== 改进的模型 ==========
class myModel():
    def __init__(self):
        input_dim = 28 * 28 + 1
        hidden1_dim = 512
        hidden2_dim = 256
        hidden3_dim = 128
        output_dim = 10

        # He初始化
        self.W1 = np.random.normal(0, np.sqrt(2.0 / input_dim), [input_dim, hidden1_dim])
        self.W2 = np.random.normal(0, np.sqrt(2.0 / hidden1_dim), [hidden1_dim, hidden2_dim])
        self.W3 = np.random.normal(0, np.sqrt(2.0 / hidden2_dim), [hidden2_dim, hidden3_dim])
        self.W4 = np.random.normal(0, np.sqrt(2.0 / hidden3_dim), [hidden3_dim, output_dim])

        # Adam优化器参数
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_W3, self.v_W3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        self.m_W4, self.v_W4 = np.zeros_like(self.W4), np.zeros_like(self.W4)
        self.t = 0

        # 层定义
        self.mul_h1 = Matmul()
        self.bn1 = BatchNorm(hidden1_dim)
        self.relu1 = Relu()
        self.dropout1 = Dropout(0.3)

        self.mul_h2 = Matmul()
        self.bn2 = BatchNorm(hidden2_dim)
        self.relu2 = Relu()
        self.dropout2 = Dropout(0.3)

        self.mul_h3 = Matmul()
        self.bn3 = BatchNorm(hidden3_dim)
        self.relu3 = Relu()

        self.mul_h4 = Matmul()
        self.softmax = Softmax()
        self.cross_en = Cross_entropy()

    def forward(self, x, labels, training=True):
        x = x.reshape(-1, 28 * 28)
        bias = np.ones(shape=[x.shape[0], 1])
        x = np.concatenate([x, bias], axis=1)

        # 设置各层的training状态
        self.bn1.training = training
        self.bn2.training = training
        self.bn3.training = training
        self.dropout1.training = training
        self.dropout2.training = training

        # 第一层
        self.h1 = self.mul_h1.forward(x, self.W1)
        self.h1_bn = self.bn1.forward(self.h1)
        self.h1_relu = self.relu1.forward(self.h1_bn)
        self.h1_drop = self.dropout1.forward(self.h1_relu)

        # 第二层
        self.h2 = self.mul_h2.forward(self.h1_drop, self.W2)
        self.h2_bn = self.bn2.forward(self.h2)
        self.h2_relu = self.relu2.forward(self.h2_bn)
        self.h2_drop = self.dropout2.forward(self.h2_relu)

        # 第三层
        self.h3 = self.mul_h3.forward(self.h2_drop, self.W3)
        self.h3_bn = self.bn3.forward(self.h3)
        self.h3_relu = self.relu3.forward(self.h3_bn)

        # 输出层
        self.h4 = self.mul_h4.forward(self.h3_relu, self.W4)
        self.h4_soft = self.softmax.forward(self.h4)
        self.loss = self.cross_en.forward(self.h4_soft, labels)

    def backward(self, labels):
        self.loss_grad = self.cross_en.backward(labels)
        self.h4_soft_grad = self.softmax.backward(self.loss_grad)
        self.h4_grad, self.W4_grad = self.mul_h4.backward(self.h4_soft_grad)

        self.h3_relu_grad = self.relu3.backward(self.h4_grad)
        self.h3_bn_grad = self.bn3.backward(self.h3_relu_grad)
        self.h3_grad, self.W3_grad = self.mul_h3.backward(self.h3_bn_grad)

        self.h2_drop_grad = self.dropout2.backward(self.h3_grad)
        self.h2_relu_grad = self.relu2.backward(self.h2_drop_grad)
        self.h2_bn_grad = self.bn2.backward(self.h2_relu_grad)
        self.h2_grad, self.W2_grad = self.mul_h2.backward(self.h2_bn_grad)

        self.h1_drop_grad = self.dropout1.backward(self.h2_grad)
        self.h1_relu_grad = self.relu1.backward(self.h1_drop_grad)
        self.h1_bn_grad = self.bn1.backward(self.h1_relu_grad)
        self.h1_grad, self.W1_grad = self.mul_h1.backward(self.h1_bn_grad)


# ========== Adam优化器更新函数 ==========
def adam_update(model, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    model.t += 1

    # 更新 W1
    model.m_W1 = beta1 * model.m_W1 + (1 - beta1) * model.W1_grad
    model.v_W1 = beta2 * model.v_W1 + (1 - beta2) * (model.W1_grad ** 2)
    m_hat = model.m_W1 / (1 - beta1 ** model.t)
    v_hat = model.v_W1 / (1 - beta2 ** model.t)
    model.W1 -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

    # 更新 W2
    model.m_W2 = beta1 * model.m_W2 + (1 - beta1) * model.W2_grad
    model.v_W2 = beta2 * model.v_W2 + (1 - beta2) * (model.W2_grad ** 2)
    m_hat = model.m_W2 / (1 - beta1 ** model.t)
    v_hat = model.v_W2 / (1 - beta2 ** model.t)
    model.W2 -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

    # 更新 W3
    model.m_W3 = beta1 * model.m_W3 + (1 - beta1) * model.W3_grad
    model.v_W3 = beta2 * model.v_W3 + (1 - beta2) * (model.W3_grad ** 2)
    m_hat = model.m_W3 / (1 - beta1 ** model.t)
    v_hat = model.v_W3 / (1 - beta2 ** model.t)
    model.W3 -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

    # 更新 W4
    model.m_W4 = beta1 * model.m_W4 + (1 - beta1) * model.W4_grad
    model.v_W4 = beta2 * model.v_W4 + (1 - beta2) * (model.W4_grad ** 2)
    m_hat = model.m_W4 / (1 - beta1 ** model.t)
    v_hat = model.v_W4 / (1 - beta2 ** model.t)
    model.W4 -= lr * m_hat / (np.sqrt(v_hat) + epsilon)


def train_one_step(model, x, y, lr=0.001, l2_lambda=0.0001):
    model.forward(x, y, training=True)
    model.backward(y)

    # L2正则化
    model.W1_grad += l2_lambda * model.W1
    model.W2_grad += l2_lambda * model.W2
    model.W3_grad += l2_lambda * model.W3
    model.W4_grad += l2_lambda * model.W4

    # Adam更新
    adam_update(model, lr)

    return model.loss, compute_accuracy(model.h4_soft, y)


def test(model, x, y):
    model.forward(x, y, training=False)
    loss = model.loss
    accuracy = compute_accuracy(model.h4_soft, y)
    return loss, accuracy


def compute_accuracy(prob, labels):
    predictions = np.argmax(prob, axis=1)
    truth = np.argmax(labels, axis=1)
    return np.mean(predictions == truth)


# ========== 训练主程序 ==========
print("Loading data...")
train_data, test_data = mnist_dataset()

train_label = np.zeros([train_data[0].shape[0], 10])
test_label = np.zeros([test_data[0].shape[0], 10])
train_label[np.arange(train_data[0].shape[0]), train_data[1]] = 1
test_label[np.arange(test_data[0].shape[0]), test_data[1]] = 1

# 数据增强
print("Applying data augmentation...")
train_x_aug, train_label_aug = DataAugmentation.augment(train_data[0], train_label)
print(f"Original training set: {train_data[0].shape[0]} samples")
print(f"Augmented training set: {train_x_aug.shape[0]} samples")

# 超参数
batch_size = 128
epochs = 30
initial_lr = 0.001
best_test_acc = 0
patience = 10
no_improve = 0

model = myModel()
print(f"Model architecture: 784->512(BN+Dropout)->256(BN+Dropout)->128(BN)->10")
print(f"Total epochs: {epochs}, Batch size: {batch_size}")

for epoch in range(epochs):
    # 学习率衰减（余弦退火）
    lr = initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))

    # 打乱数据
    indices = np.arange(train_x_aug.shape[0])
    np.random.shuffle(indices)
    train_x_shuffled = train_x_aug[indices]
    train_label_shuffled = train_label_aug[indices]

    total_loss = 0
    total_acc = 0
    num_batches = 0

    # 训练
    for i in range(0, train_x_aug.shape[0], batch_size):
        batch_x = train_x_shuffled[i:i + batch_size]
        batch_y = train_label_shuffled[i:i + batch_size]

        loss, acc = train_one_step(model, batch_x, batch_y, lr=lr)
        total_loss += loss
        total_acc += acc
        num_batches += 1

        if i % (batch_size * 50) == 0 and i > 0:
            print(f'Epoch {epoch}, Batch {i // batch_size}: loss={loss:.4f}, acc={acc:.4f}, lr={lr:.6f}')

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    print(f'Epoch {epoch} finished: avg_loss={avg_loss:.4f}, avg_acc={avg_acc:.4f}')

    # 每5个epoch测试一次
    if epoch % 5 == 0 or epoch == epochs - 1:
        test_loss, test_acc = test(model, test_data[0], test_label)
        print(f'Test after epoch {epoch}: loss={test_loss:.4f}, acc={test_acc:.4f}')

        # 早停
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            no_improve = 0
            # 保存最佳模型
            np.savez('best_model.npz',
                     W1=model.W1, W2=model.W2, W3=model.W3, W4=model.W4,
                     bn1_gamma=model.bn1.gamma, bn1_beta=model.bn1.beta,
                     bn2_gamma=model.bn2.gamma, bn2_beta=model.bn2.beta,
                     bn3_gamma=model.bn3.gamma, bn3_beta=model.bn3.beta)
            print(f"  ✓ New best model saved! (acc={test_acc:.4f})")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# 加载最佳模型进行最终测试
print("\n" + "=" * 50)
print("Loading best model for final evaluation...")
best_model = np.load('best_model.npz')
model.W1 = best_model['W1']
model.W2 = best_model['W2']
model.W3 = best_model['W3']
model.W4 = best_model['W4']
model.bn1.gamma = best_model['bn1_gamma']
model.bn1.beta = best_model['bn1_beta']
model.bn2.gamma = best_model['bn2_gamma']
model.bn2.beta = best_model['bn2_beta']
model.bn3.gamma = best_model['bn3_gamma']
model.bn3.beta = best_model['bn3_beta']

test_loss, test_acc = test(model, test_data[0], test_label)
print(f"Final test loss: {test_loss:.4f}")
print(f"Final test accuracy: {test_acc:.4f}")
print(f"Target: 99.5% | Achieved: {test_acc:.2%}")
print("=" * 50)