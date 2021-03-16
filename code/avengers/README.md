# Avengers

* model1 : Eff2020
* model2 : Eff2019+2020
* model3 : Reg2020
* model4 : Reg2019+2020
* model5 : Eff2020 (seed 720)
* model6 : VIT
* model7 : FixMatch
* model8 : Distillation
* model9 : NFNet
* model10 : Reg nocv


### model10 is refered to Reg2020. I didn't get the code.

## Result  (Public 13nd Private 184nd)

|   submission   | Public LB | Rank | Private LB | Rank |
| :------------: | :-------: | :--: | :--------: | :--: |
| EfficientNetB4 |   0.905   |      |   0.896    |      |
|   RegNetY_40   |   0.905   |      |   0.895    |      |
|   VIT_16_384   |   0.902   |      |   0.894    |      |
|  Distillation  |     -     |  -   |     -      |  -   |
|     NFNet      |   0.899   |      |   0.894    |      |
|     FixMatch   |   0.9005  |      |   0.8987   |      |
|   Ensemble1    |   0.9048  |      |   0.8993   | 171  |
|   Ensemble2    |   0.9082  |  13  |   0.8975   |      |

* Ensemble1 = Avengers
* Ensemble2 = Effx3 + Regx2 + Reg-Distill 
