import keras
from model import define_model
from callbacks import CustomModelCheckoint

model = define_model()

callbacks = CustomModelCheckoint()

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    verbose=1,
    callbacks=callbacks,
    workers=4,
    max_queue_size=8,
)
