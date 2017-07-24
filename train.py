import chainer
import chainer.functions as F
import chainer.iterators as I
import chainer.links as L
import chainer.optimizers as O
from chainer import training
import chainer.training.extensions as E


import model2
import load_data_ecfp


C = 12
nfp_hidden_dim = 50
nfp_out_dim = 120
max_degree = 5
radius = 6
mlp_hid_dim = 100
batchsize = 100
epoch = 20

train, val, test, atom2id = load_data_ecfp.make_dataset()
nfp = model2.NFP(nfp_hidden_dim, nfp_out_dim, max_degree, len(atom2id), radius)

predictor = model2.Predictor(nfp, mlp_hid_dim, C)
model = L.Classifier(predictor,
                     lossfun=F.sigmoid_cross_entropy,
                     accfun=F.binary_accuracy)

optimizer = O.Adam()
optimizer.setup(model)

train_iter = I.SerialIterator(train, batchsize)
val_iter = I.SerialIterator(val, batchsize, repeat=False, shuffle=False)
test_iter = I.SerialIterator(test, batchsize, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (epoch, 'epoch'))

eval_model = model.copy()
eval_nfp = eval_model.predictor


trainer.extend(E.LogReport(trigger=(2, 'iteration')))
trainer.extend(E.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy','elapsed_time']))
trainer.extend(E.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(E.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
#trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
#trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
trainer.extend(E.Evaluator(val_iter, model),
               trigger=(2, 'iteration'))
trainer.extend(E.dump_graph('main/loss'))

trainer.run()

print('test')
evaluator = E.Evaluator(test_iter, eval_model)
result = evaluator()
print('valid accuracy:', float(result['main/accuracy']))

# save model
chainer.serializers.save_npz('model.npz', model)


