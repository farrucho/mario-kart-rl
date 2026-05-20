## sources:
https://stable-retro.farama.org/getting_started/
https://stable-retro.farama.org/integration/#supported-roms



## variaveis da ram do jogo:
https://datacrystal.tcrf.net/wiki/Super_Mario_Kart/RAM_map
https://bin.smwcentral.net/u/34395/SMK_Potential_Ram_Addresses.txt

coins 
speed_east
speed_south
speed_overall esta é inutil pois é a soma da east + south, se o carro andar para noroeste é tudo negativo, oque nao faz sentido, usar o modulo da velocidade ao inves desta
surface type, util para ver se esta a andar fora da estrada
clock nao é cronometro
checkpoint vai de 0-X, e vai dando reset por volta

end_condition: if player finished (lap number == 133) OR all racers finished (racers_finished >= 14 )


## reward function prototype:
recompensar:
+ laps feitas
+ posicao real time
- posicao final
+ new checkpoint reached
+ tempo por lap

```
R(s) = (when lap is increased)*100 + 5*1/rank + 5*(when checkpoint is increased) - -0.1*time
```
