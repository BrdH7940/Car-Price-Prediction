<b>Gradient descent cho Mean square error</b>
Chúng ta biết rằng thuật toán gradient cho mean square error là 
$$x(t+1) = x(t) - \alpha\frac{2U^T(Ux(t)-Y)}{m} \quad t\in \mathbb{N}$$
Như đã chứng minh ở phần trước , hàm $MSE$ là hàm lồi nên chỉ có tối đa 1 điểm cực trị, điểm cực trị nếu có thì là điểm cực tiểu toàn cục và là duy nhất