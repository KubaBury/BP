using Plots
using Distributions
using LinearAlgebra
using LaTeXStrings


## Generate data,
u = Uniform(-1,2)
x = rand(u,300)
y = x.^2 .+ randn(length(x),1)
p1 = plot(x,y, seriestype = :scatter, xlabel="x", ylabel="y", label ="data", legend=:topleft)

#p(x,y) = p(y|x)*p(x)


#p_hx=Plots.histogram(x,normalize=:pdf, nbins = 40,label="normalized histogram", xlabel="x", ylabel = "normalized counts" );
# x ~ U(-1,2)
#U = Uniform(-1, 2) #p(x) = 1/3 na (-1,2) otherwise 0
#plot!(x, pdf.(U, x), lw = 5, label = "U(-1,2)")
#OLS
W = [(x).^i for i = 0:2];
X = hcat(W...);
θ = inv(X'*X)*X'*y;
#  assume noise ε_i ~ N(0,1) 
#p(y|x) = N(Xθ,I)  σ^2 = 1 
B = X*θ ; 

plt(k,l) =pdf(Uniform(-1, 2), k) * pdf.(Normal.(B[1], 1.), l) 
contour!(p1,-1.5:0.02:2.5,-2:0.02:7.0,(k,l)->plt(k,l), xlabel=L"x", ylabel=L"y", levels=10 )
