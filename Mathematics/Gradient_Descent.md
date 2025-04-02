<b style  = "font-size:25px">Thuật toán Gradient descent</b>
Ở trong phần này chúng ta giả sử rằng các hàm số có đạo hàm liên tục đến cấp 2 tại mọi điểm thuộc tập xác định của nó
<b> 1. Vector gradient</b><br>
đặt $X =(x_1,x_2,x_3,...x_n)$ xét hàm số $f(X) = f(x_1,x_2,...x_n)$ $n$ biến $x_1,x_2,...x_n$ vector gradient của $f$ là vector gồm các thành phần là đạo hàm riêng của $f$ , kí hiệu là $\nabla f$ , với : 
$$\nabla f = \begin{pmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots\\
\frac{\partial f}{\partial x_n}\\
\end{pmatrix}$$

<b> 2. Đạo hàm theo hướng</b><br>
giả sử ta muốn tính đạo hàm theo hướng của hàm $f$ tại $X = X^{*} = (x_1^*,...x_n^{*})$  theo hướng của vector $D = (d_0,...d_n)$ , khi đó ta sẽ tính đạo hàm của hàm số $f(X^{*}+tD)$ tại $t = 0$ , ta có 
$$\frac{\partial f(X^{*}+tD)}{\partial t} = \nabla f(X^*).D  = \nabla f(X^{*})^TD$$


<b> 3. Tính chất đạo hàm theo hướng</b><br>


ta có $\nabla f(X^*).D \leq \|\nabla f(X^*)\|.\|D\|$ đẳng thức xảy ra khi $D$ cùng hướng với vector gradient <br>
ta có $\nabla f(X^*).D \geq -\|\nabla f(X^*)\|.\|D\|$ đẳng thức xảy ra khi $D$ ngược hướng với vector gradient <br>
Như vậy giá trị của hàm số tăng mạnh nhất theo hướng vector gradient, vì đạo hàm theo hướng vector gradient có giá trị dương nên khi di chuyển dọc theo hướng vector gradient trong 1 khoảng nhỏ thì giá trị của hàm số tăng <br>

giá trị của hàm số giảm mạnh nhất theo hướng ngược hướng vector gradient, vì đạo hàm theo hướng  ngược hướng vector gradient có giá trị âm nên khi di chuyển dọc theo hướng ngược hướng vector gradient trong 1 khoảng nhỏ thì giá trị của hàm số giảm <br>
nếu f đạt cực trị địa phương hoặc cực trị toàn cục tại $X^{*}$ thì $\nabla f(X^{*}) = 0$

<b> 4. Thuật toán gradient descent</b><br>
Thuật toán gradient descent là thuật toán dc sử dụng để tìm điểm cực tiểu của 1 hàm cho trước, tại đó giá trị của hàm số là giá trị nhỏ nhất nghĩa là thuật toán này nhằm cực tiểu hóa 1 hàm số cho trước tức là tìm điểm mà tại đó gradient bằng $0$. <br>
Thuật toán này dựa trên tính chất của vector gradient , di chuyển ngược hướng $\nabla f$ thì giá trị hàm số giảm, di chuyển cùng hướng $\nabla f$ thì giá trị hàm số tăng, đặt 
$g(u) = \frac{\partial f(X^{*}+tD)}{\partial t}(t = u)$ , giả sử $g(0) >0$ , khi đó $\exists \epsilon > 0 : u \in \left [-\epsilon , \epsilon\right] \Rightarrow g(u) \geq 0$ , điều này có nghĩa là hàm số đồng biến trong 1 khoảng nhỏ <br>
Giả sử chúng ta bắt đầu tại điểm $x(0)\in R^n$ khi đó thuật toán gradient có công thức như sau 


$$x(t+1) = x(t) - \alpha\nabla f(x(t)) \quad t\in \mathbb{N} (*)$$


Ở đây $\alpha$ là 1 số thực dương dc gọi là hệ số học <br>
Đầu tiên chúng ta chứng minh rằng nếu $x(t)$ hội tụ thì nó sẽ hội tụ tại điểm mà $\nabla f = 0$ , giả sử $\lim x(t) = x^{*}$ khi đó ta có phương trình $x^{*} = x^{*}-\alpha\nabla f(x^{*})$ hay $\nabla f(x^{*}) = 0$ , điều này đảm bảo rằng $x(t)$ hội tụ điến điểm cực trị nếu nó hội tụ<br>
giả sử $f$ có cực tiểu toàn cục và dãy $f(x(t))$ giảm kể từ 1 giá trị nào đó , khi đó dãy $f(x(t))$ là dãy giảm và bị chặn dưới nên hội tụ nên $x(t)$ hội tụ , và nó sẽ hội tụ về điểm cực tiểu của $f$
<b> Ý nghĩa của hệ số $\alpha$</b>
đặt $G(X) = \nabla f(X)$ , xét điểm $X^{*}$ bất kì sao cho $G(X^{*})\neq 0$, đặt $q(u) = \frac{\partial f(X^{*}-G(X^{*})t)}{\partial t}(t = u)$ , đây chính là đạo hàm ngược hướng vector gradient tại $X^{*}$ ta có $q(0) < 0$ hàm số $q(u)$ liên tục tại mọi điểm thuộc lân cận $u = 0$ do đó tồn tại $\epsilon > 0$ sao cho $q(u)<0 \quad \forall u \in \left [ -\epsilon,\epsilon \right ]$ , điều này có nghĩa hàm số $p(u) = f(X^{*}-G(X^{*})u)$ nghịch biến trên $\left [-\epsilon, \epsilon \right]$ suy ra $p(0) < p(\epsilon)$ hay $f(X^{*}-\nabla f(X^{*})\epsilon) <f(X^{*})$ , hệ số $\alpha$ ở trên $(*)$ có vai trò giống như $\epsilon$ <br>
gọi $\epsilon^{*}$ là số thực dương lớn nhất sao cho $p(u)\leq 0 \quad \forall u\in \left[0 ,\epsilon^{*}\right]$ khi đó nếu $\alpha \leq \epsilon^{*}$ và $\alpha > 0$ thì ta luôn có $f(X^{*}-\alpha\nabla f(X^{*}))< f(X^{*})$, như vậy nếu ta chọn $\alpha$ thích hợp sao cho $f(x(t))$ luôn giảm thì dãy $f(x(t))$ sẽ hội tụ về giá trị cực tiểu địa phương hoặc cực tiểu toàn cục của $f$ hay $x(t)$ sẽ hội tụ về điểm cực tiểu của $f$