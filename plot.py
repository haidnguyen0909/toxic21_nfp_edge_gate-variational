import pickle
import matplotlib.pyplot as plt
import numpy as np

# net 1: variational 100+100 nfp, 100+100+100 for mlp
# for training
with open("variational_train_2x100nfp_3x100mlp_rec", "rb") as fp:
    rec_train_loss = pickle.load(fp)
#with open("variational_train_2x100nfp_3x100mlp_kl", "rb") as fp:
#    kl_train_loss = pickle.load(fp)
with open("variational_train_2x100nfp_3x100mlp_loss", "rb") as fp:
    train_loss = pickle.load(fp)

# for valid
with open("variational_valid_2x100nfp_3x100mlp_rec", "rb") as fp:
    rec_valid_loss = pickle.load(fp)
with open("variational_valid_2x100nfp_3x100mlp_kl", "rb") as fp:
    kl_valid_loss = pickle.load(fp)
with open("variational_valid_2x100nfp_3x100mlp_loss", "rb") as fp:
    valid_loss = pickle.load(fp)

# for test
#with open("variational_test_2x100nfp_3x100mlp_rec", "wb") as fp:
#    pickle.dump(rec_test_loss, fp)
#with open("variational_test_2x100nfp_3x100mlp_kl", "wb") as fp:
#    pickle.dump(rec_test_loss, fp)
#with open("variational_test_2x100nfp_3x100mlp_acc", "wb") as fp:
#    pickle.dump(accuracy, fp)


fig = plt.figure()
ax = plt.axes()

x = np.arange(40)
#plt.plot(x, np.array(rec_train_loss), color = 'blue', linestyle = 'solid', label = "rec_train")
plt.plot(x, np.array(rec_valid_loss), color = 'red', linestyle = 'solid', label = "rec_valid_loss")
plt.plot(x, np.array(valid_loss), color = 'blue', linestyle = 'dotted', label = "valid loss")
plt.plot(x, np.array(kl_valid_loss), color = 'red', linestyle = 'dashed', label = "kl_valid_loss")

plt.title("training error")
plt.xlabel("epoch")
plt.ylabel("training error")
plt.axis("tight")
plt.legend()


plt.show()
