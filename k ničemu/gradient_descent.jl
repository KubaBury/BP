using Flux

W = rand(2, 5);
b = rand(2);

predict(x) = W*x .+ b;

function loss(x, y)
  ŷ = predict(x)
  sum((y .- ŷ).^2)
end

x, y = rand(5), rand(2) ;
L1 = loss(x, y) ;

gs = gradient(() -> loss(x, y), params(W, b));
#Now that we have gradients, we can pull them out
# and update W to train the model.

W̄ = gs[W];
W .-= 0.1 .* W̄;
L2 = loss(x, y); 
L1,L2