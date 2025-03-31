<b style = "font-size:20px">Cơ sở toán học</b> 
Giả sử chúng ta có dữ liệu có các cột đầu vào $X_1,X_2,....X_n$ và cột đầu ra $Y$ , kí hiệu $X_i(j)$ là giá trị  hàng $j$ cột $i$ trong bảng dữ liệu  , giả sử dữ liệu có m hàng, ta có : 
$$X_i =  \begin{pmatrix}
 &X_i(1)\\
 &X_i(2) \\ 
 &\vdots\\
 &X_i(m)
\end{pmatrix} (1)$$

chúng ta muốn xấp xỉ $Y$ bởi 1 hàm tuyến tính theo các cột đầu vào, giả sử  $Y = b+ \epsilon +\sum_{i=1}^{n} a_iX_i$ , với $\mathbb{E}\left [ \epsilon \right  ] = \bar{\epsilon} = 0$ và phương sai hữu hạn , đặt $v_n$ là vector cột $n$ chiều có tất cả các phần tử bằng $1$  ,ta có 
$$Y-\sum_{i=1}^{n}a_iX_i - bv_{m} = \epsilon$$
 đặt $X = \sum_{i=1}^{n}a_iX_i + bv_{m}$,viết lại dưới dạng ma trận  đặt $U = (X_1,X_2,...X_n,v_m)$ và $v = (a_1,x_2,...a_n,b)$  ,  ta có $X = Uv$ và :
$$ Y-X = \epsilon$$

chúng ta sẽ tìm siêu phẳng $n+1$ chiều sao cho phương sai của $\epsilon$ là nhỏ nhất , ta có:
$$\sigma^2(\epsilon) = \frac{1}{m}\sum_{i=1}^m\epsilon^2(i) = \frac{1}{m}\sum_{i=1}^{n} (Y(i)-\sum_{i=1}^{n}a_iX_i - b)^2 $$
điều nay tương đương với tìm các hệ số $a_1,a_2,...,a_n,b$ sao cho $\frac{\|Y-X\|^2}{m}$ là nhỏ nhất hay $\|Uv-Y \|^2$ nhỏ nhất , dưới góc nhìn hình học , gọi $S$ là không gian sinh bởi các vector $X_1,X_2,...X_n,v_m$ điều này tương đương với tìm vector $X \in S$ sao cho khoảng cách từ $X$ đến $Y$ là nhỏ nhất <br>
<b>Định nghĩa hình chiếu :</b> xét $V$ là  1 không gian vector con của $\mathbb{R}^n$ và $u \in \mathbb{R^n}$ ,hình chiếu của $u$ lên $V$ là vector $v\in V$ thỏa mãn $x.(u-v) = 0 \quad (*) \quad \forall  x\in V$ kí hiệu là $proj_V(u)$
<b>Tính chất : </b> hình chiếu là duy nhất , thật vậy, giả sử $v,v_1 \in V$ thỏa mãn $(*)$ đặt $v_1 = v+v_0 \quad (v_0 \in V)$  ta có $x.(u-v)=0$ và $x.(u-v_1)=0$ hay $x.(v-v_1)=0$ tương đương với $x.v_0 = 0$ hay $v_0 \in V^{\perp}$ mà $v_0 \in V$ suy ra $v\in V\cap V^{\perp} = \{0\}$ hay $v_0 = 0$ suy ra $v_1 = v$ suy ra dpcm 

<b> Tính hình chiếu  </b>  
gọi $v_1,v_2,...,v_m \in \mathbb{R}^n$ là cơ sở của không gian vector $V$ , và $u \in \mathbb{R}^n$ chúng ta cần tính hình chiếu của $u$ lên $V$ , đặt $S = (v_1,v_2,...v_m)$ là ma trận có các cột là các vector $v_1,v_2,..v_m$ , khi đó mọi vector trong $V$ có dạng $S x$ với $x \in \mathbb{R}^m$  , khi đó hình chiếu của $u$ lên $V$ thỏa mãn 
$$S^T(Sx-u) = 0   \Leftrightarrow  S^TSx= S^Tu  \Leftrightarrow  x = (S^TS)^{-1}S^Tu \quad (2)$$ 
hay $Sx = S(S^TS)^{-1}S^Tu$
vậy ma trận chiếu 1 vector lên $V$ là $S(S^TS)^{-1}S^T$<br> 

quay trở lại bài toán , chúng ta tìm vector $X \in U$ sao cho  $\|X-Y\|^2$ nhỏ nhất, gọi $p$ là hình chiếu của $Y$ lên $U$ , ta có $(Y-p)^T.z = 0  \quad \forall z  \in U$ , đặt $k = Y-p$ ta có $k^T.z = 0 \quad \forall z  \in U$ ta chứng minh $\|p-Y\|^2$ nhỏ nhất, $\forall u\in U$ đặt $u = p+u_0$ , ta có $\|u-Y\|^2  =  ||u-p+p-Y \|^2 = ||u_0+z\|^2 = \|u_0\|^2 + \|z\|^2$ do $u_0.z = 0$, ta có $||u_0+z\|^2 = \|u_0\|^2 + \|z\|^2 \geq \|z\|^2$, đẳng thức xảy ra $\Leftrightarrow \|u_0\|^2 = 0 \Leftrightarrow \ u_0 = 0$ , vậy  $\|X-Y\|^2$ đạt dc giá trị nhỏ nhất  $\Leftrightarrow X = p$ , vậy hệ số cần tìm là $x = U(U^TU)^{-1}U^TY$ 
<b>*Chứng minh $U^TU$ khả nghịch</b>
vì $U$ có các cột độc lập tuyến tính nên $\forall x\in \mathbb{R}^{n+1}$ ta có $Ux = 0 \Leftrightarrow x = 0$ , giả sử $x_0$ thỏa mãn $U^TUx_0=0$ , đặt $Y_0 = Ux_0$ ta có $Y_0 \in U$ và $UY_0 = 0$ , hay $Y_0$ trực giao với các vector cơ sở của $U$ hay $Y_0$ trực giao với $U$ , suy ra $Y_0 \in U^{\perp}$ , suy ra $Y_0 \in U\cap U^{\perp} = \{0\} \Leftrightarrow Y_0 = 0\Leftrightarrow Ux_0 = 0 \Leftrightarrow x_0 = 0 \Rightarrow (U^TUx_0 = 0 \Leftrightarrow x_0 = 0 )$ suy ra $U^TU$  khả nghịch 

<b style = "font-size:20px">Hồi quy tuyến tính</b>


Chúng ta cần 1 cơ sở để kiểm tra xem liệu mô hình hồi quy tuyến tính có phù hợp không, giả sử khi dữ liệu không có nhiễu khi đó $Y = \sum_{i=1}^{n}a_iX_i + bv_m (3)$ , vì $v_m$ là cột có tất cả các hàng đều bằng nhau nên chúng ta sẽ bỏ qua nó , trong $(3)$ lấy trung bình cả 2 vế ta có $\bar{Y} = \sum_{i=1}^{n}a_i\bar{X_i} + b$ hay $$\bar{Y}b_m = \sum_{i=1}^{n}a_i\bar{X_i}v_m + b.v_m \Leftrightarrow Y - v_m\bar{Y} = \sum_{i=1}^{n}(a_iX_i-a_i\bar{X_i}v_m) + b.v_m-b.v_m$$ hay $Y - b\bar{Y} = \sum_{i=1}^{n}(a_iX_i-a_i\bar{X_i}v_m)$ <br> với mọi $x \in R^m$ đặt $x^{*} = x-\bar{x}v_m$ ta có $Y^{*} = \sum_{i=1}^n a_iX_{i}^{*}$ , trong thực tế dữ liệu luôn có  nhiễu nên $Y \neq \sum_{i=1}^n a_iX_{i}^{*}$ , chúng ta sẽ đo sự tương quan tuyến tính của $Y^{*}$ với $X_1^{*},X_2^{*},..X_n^{*}$ <br>
Nhớ lại mô hồi quy tuyến tính đơn biến $y = ax +b$ hay $y^{*} = ax^{*}$ , chúng ta có công thức hệ số tương quan $r = \frac{\sum_{i=1}^m x^{*}_iy^{*}_i}{\|x^{*}\|*\|y^{*}\|}$ là cosin giữa 2 vector $x^{*}$ và $y^{*}$ , tương tự với mô hình hồi quy tuyến tính đa biến chúng ta sẽ tính cosin góc giữa $Y$ với không gian $D$ sinh bởi $X_1^{*},X_2^{*},...X_n^{*}$ , chính là cosin giữa hình góc giữa hình chiếu của $Y$ lên không gian $D$ và $Y$ , đặt $P = proj_D(Y)$ , ta có $r = \frac{\|P\|}{\|Y\|}$ <br>
Minh họa với n = 1, n = 2







 