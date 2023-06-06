SET dataset=MNIST-CIFAR
SET type=SimilarNet
SET normalize=True

@echo off

SET /A seq = 0

:seq_loop
IF EXIST counter\%seq% (
SET /A seq = seq + 1
goto seq_loop
)

IF %seq% LSS 20 (
@echo on
mkdir counter\%seq%
python Experiment_%dataset%.py %type% %normalize%
shutdown -r -t 60
)

@echo on
