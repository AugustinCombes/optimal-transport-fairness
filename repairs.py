import numpy as np
import ot
from disparate import *
from utils import simulate_dataset, format_dataset
from sklearn.linear_model import LogisticRegression

def geometric_repair(X0, X1, lmbd):

    Ae_01, be_01 = ot.da.OT_mapping_linear(X0, X1)
    Ae_10, be_10 = ot.da.OT_mapping_linear(X1, X0)

    w0,w1 = X0.shape[0], X1.shape[1]
    w0,w1 = w0/(w0+w1),w1/(w0+w1)

    barycenter_0 = w1*(X0.dot(Ae_01) + be_01) + w0*X0
    barycenter_1 = w0*(X1.dot(Ae_10) + be_10) + w1*X1

    X0_repaired = lmbd*(barycenter_0) + (1-lmbd)*X0
    X1_repaired = lmbd*(barycenter_1) + (1-lmbd)*X1
    return  X0_repaired, X1_repaired

def DI_list_geometric_repair(X0, X1):
    
    Ae_01, be_01 = ot.da.OT_mapping_linear(X0, X1)
    Ae_10, be_10 = ot.da.OT_mapping_linear(X1, X0)

    def geometric_repair(X0, X1, lmbd):
        w0,w1 = X0.shape[0], X1.shape[1]
        w0,w1 = w0/(w0+w1),w1/(w0+w1)

        barycenter_0 = w1*(X0.dot(Ae_01) + be_01) + w0*X0
        barycenter_1 = w0*(X1.dot(Ae_10) + be_10) + w1*X1

        X0_repaired = lmbd*(barycenter_0) + (1-lmbd)*X0
        X1_repaired = lmbd*(barycenter_1) + (1-lmbd)*X1
        return  X0_repaired, X1_repaired

    lambdas = np.linspace(0,1,1000)
    DIs = []

    beta0 = (1, -1, -0.5, 1, -1)
    beta1 = (1, -0.4, 1, -1, 1)

    for lmbd in lambdas:
        X0r, X1r = geometric_repair(X0, X1, lmbd)
        # print(X0r.sum())

        y0 = np.exp(X0r.dot(beta0)) / (1 + np.exp(X0r.dot(beta0)))
        y1 = np.exp(X1r.dot(beta1)) / (1 + np.exp(X1r.dot(beta1)))
        X,Y = format_dataset(X0r, X1r, y0, y1)

        # X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)

        # print((Y>0.5).astype(int))

        clf = LogisticRegression(random_state=69).fit(X[:, 1:],(Y>0.5).astype(int))
        Y_pred = clf.predict(X[:, 1:])

        DIs.append(disparate(X,Y_pred,0)[1])
    return DIs

def random_repair(X0, X1, lmbd):

    Ae_01, be_01 = ot.da.OT_mapping_linear(X0, X1)
    Ae_10, be_10 = ot.da.OT_mapping_linear(X1, X0)

    w0,w1 = X0.shape[0], X1.shape[1]
    w0,w1 = w0/(w0+w1),w1/(w0+w1)

    barycenter_0 = w1*(X0.dot(Ae_01) + be_01) + w0*X0
    barycenter_1 = w0*(X1.dot(Ae_10) + be_10) + w1*X1

    ber0, ber1 = np.random.binomial(1, lmbd, size=(X0.shape[0], 1)), np.random.binomial(1, lmbd, size=(X1.shape[0], 1))

    X0_repaired = ber0*(barycenter_0) + (1-ber0)*X0
    X1_repaired = ber1*(barycenter_1) + (1-ber1)*X1
    return  X0_repaired, X1_repaired

def DI_list_random_repair(X0, X1):
    
    Ae_01, be_01 = ot.da.OT_mapping_linear(X0, X1)
    Ae_10, be_10 = ot.da.OT_mapping_linear(X1, X0)

    def random_repair(X0, X1, lmbd):
        w0,w1 = X0.shape[0], X1.shape[1]
        w0,w1 = w0/(w0+w1),w1/(w0+w1)

        barycenter_0 = w1*(X0.dot(Ae_01) + be_01) + w0*X0
        barycenter_1 = w0*(X1.dot(Ae_10) + be_10) + w1*X1

        ber0, ber1 = np.random.binomial(1, lmbd, size=(X0.shape[0], 1)), np.random.binomial(1, lmbd, size=(X1.shape[0], 1))

        X0_repaired = ber0*(barycenter_0) + (1-ber0)*X0
        X1_repaired = ber1*(barycenter_1) + (1-ber1)*X1
        return  X0_repaired, X1_repaired

    lambdas = np.linspace(0,1,200)
    DIs = []

    beta0 = (1, -1, -0.5, 1, -1)
    beta1 = (1, -0.4, 1, -1, 1)

    for lmbd in lambdas:
        X0r, X1r = random_repair(X0, X1, lmbd)
        # print(X0r.sum())

        y0 = np.exp(X0r.dot(beta0)) / (1 + np.exp(X0r.dot(beta0)))
        y1 = np.exp(X1r.dot(beta1)) / (1 + np.exp(X1r.dot(beta1)))
        X,Y = format_dataset(X0r, X1r, y0, y1)

        # X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)

        # print((Y>0.5).astype(int))

        clf = LogisticRegression(random_state=69).fit(X[:, 1:],(Y>0.5).astype(int))
        Y_pred = clf.predict(X[:, 1:])

        DIs.append(disparate(X,Y_pred,0)[1])
    return DIs