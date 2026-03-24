import numpy as np
from sklearn.preprocessing import MinMaxScaler

def embaralhar(dados):
    np.random.shuffle(dados)
    return dados
    
def separar_dados(dados, percentual_treino=0.8):
    T = len(dados)
    T_tr = int(T*percentual_treino)

    data_tr = dados[:T_tr]
    data_ts = dados[T_tr:-1]
    return data_tr, data_ts

def taxa_acerto(ob, y):
    count = np.sum(ob == y.reshape(ob.shape))
    return (count/len(y) * 100)

class Camada:
    def __init__(self, num_entradas, num_neuronios, tx_ap=0.05):
        self.num_entradas = num_entradas
        self.num_neuronios = num_neuronios
        self.tx_ap = tx_ap

        self.W = np.random.randn(num_entradas, num_neuronios)
        self.b = np.random.randn(1,num_neuronios)
    
    def ativacao(self, x):

        return self.sigmoid(x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoid(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.X = X
        soma = X @ self.W + self.b
        self.Y = self.sigmoid(soma)
        return self.Y

    def backward(self, erro):
        delta = erro * self.derivada_sigmoid(self.Y)
        erro_derivada = delta @ self.W.T
        self.W = self.W + self.tx_ap * self.X.T @ delta
        self.b = self.b + self.tx_ap * np.sum(delta, axis=0,keepdims=True)
        return erro_derivada


class MLP:
    def __init__(self, num_entradas=2,num_classes=2, tx_ap=0.05):
        self.num_entradas = num_entradas
        self.num_classes = num_classes
        self.tx_ap = tx_ap
        self.num_camadas = 0

        self.camadas = []

    def adicionar_camada(self, num_neuronios):
        if self.num_camadas == 0:
            camada = Camada(num_entradas=self.num_entradas, num_neuronios=num_neuronios)
        else:
            n = self.camadas[-1].num_neuronios
            camada = Camada(num_entradas=n, num_neuronios=num_neuronios)
        camada.tx_ap = self.tx_ap
        self.camadas.append(camada)
        self.num_camadas += 1

    def adicionar_camada_saida(self):
        self.adicionar_camada(self.num_classes)

    def forward(self, X):
        entrada = X
        saida = []
        for camada in self.camadas:
            saida = camada.forward(entrada)
            entrada = saida

        return saida

    def backward(self, erro):
        for camada in reversed(self.camadas):
            erro_derivada = camada.backward(erro)
            erro = erro_derivada

    def treinamento(self, dados, desejado, epocas=500):
        for _ in range(epocas):
            y = self.forward(dados)
            d = self.format_resposta(desejado,y)
            erro = d - y
            self.backward(erro)

    def format_resposta(self,desejado,y):
        if self.camadas[-1].num_neuronios==1:
            return desejado

        d = np.zeros(np.shape(y))
        for i in range(len(desejado)):
            d[i, desejado[i].astype(int)] = 1
        return d

    def predict(self, X):
        y = self.forward(X)
        return (y>0.5).astype(int) if self.camadas[-1].num_neuronios == 1 \
            else np.argmax(y, axis=1)


if __name__ == '__main__':
    scaler = MinMaxScaler()
    num_features = 4
    num_classes = 3
    docs_way = 'docs\chatgpt_base_perceptron_3entradas_200amostras.csv'

    data = np.loadtxt(docs_way, delimiter=',',skiprows=1)
    data = embaralhar(data)

    # print(data.shape)

    data_tr, data_ts = separar_dados(data)
    X = scaler.fit_transform(data_tr[:, :num_features])
    y = data_tr[:, -1].reshape(-1, 1)  # transforma em vetor coluna

    mlp = MLP(num_entradas=num_features, num_classes=num_classes, tx_ap=0.01)
    mlp.adicionar_camada(2)
    mlp.adicionar_camada_saida()
    mlp.treinamento(X, y, 500)

    # Testando o modelo tr
    x_ts = scaler.transform(data_tr[:, :num_features])
    y_ts = data_tr[:, -1].reshape(-1, 1)
    #testando modelo ts
    # x_ts = scaler.transform(data_ts[:, :num_features])
    # y_ts = data_ts[:, -1].reshape(-1, 1)

    ob = mlp.predict(x_ts)

    print(f"Resposta obtida: {ob}")
    print(f"Resposta esperada: {y_ts.T}\n")
    print(f"Taxa de acerto: {taxa_acerto(ob,y_ts):.2f}%\n")