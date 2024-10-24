/1024
![Screenshot from 2024-10-24 19-08-40](https://github.com/user-attachments/assets/096f84b7-a688-476e-81f8-3793b45d86e3)
這是經過A* 優化過後的路徑   但因為看的距離超出牆壁的關係  會導致覺得牆另一側的點比現在的好導致pure pursuit在規劃接下來的路徑點時想要往右邊走而撞上牆壁  
所以應該要項辦法讓機器在同一點錯誤一定次數時自動的不要那麼相信pure pursuit 自己探索出一條安全的路
