/1024

149行 要自己改一下圖片路徑  並確定是否有png圖片  png要拿來算A*

新增 and 改變
新增A*算法在340行那邊
pure pursuit有改一下取最近點的方法  不是單純的算距離找最近的點  而是以機器目前的yaw散射出去的方向找最近點
calculate_reward 那邊我把一些不必要的東西註解 我覺得那個會影響機器人的判斷

![Screenshot from 2024-10-24 19-08-40](https://github.com/user-attachments/assets/096f84b7-a688-476e-81f8-3793b45d86e3)
這是經過A* 優化過後的路徑   但因為看的距離超出牆壁的關係  會導致覺得牆另一側的點比現在的好導致pure pursuit在規劃接下來的路徑點時想要往右邊走而撞上牆壁  
所以應該要項辦法讓機器在同一點錯誤一定次數時自動的不要那麼相信pure pursuit 自己探索出一條安全的路

明天弄讓深度學習知道什麼時候要接管action( action目前是對應到pure pursuit所以執行結果都一樣）
reward的計算目前怪怪的  在step裡面有算  在calculate_reward裡面也有算 不知道哪些有沒有重複到

==============================================================================================
