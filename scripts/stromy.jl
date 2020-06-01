using Distributions
using Statistics
using Plots
using LaTeXStrings


i = 10000 #celkovy pocet y_i
λ = 10 #rate in Poisson distribution
P = Poisson(λ)
U = Uniform(0,10)
pocet_x = rand(P,i) #pocty x_k prirazene k jednomu y_i
k = sum(pocet_x) #celkovy pocet zaznamu na x
x = rand(U, k)

y = [mean(x[1:pocet_x[j]]) for j = 1:i] #mean of 
s = var(y)
#p(y,\bar{x})=p(\bar{x})*p(y|\bar{x})



p_y=Plots.histogram(y, nbins = 100, xlabel="x", ylabel = "counts" );
display(p_y)
plt(k,l) = pdf.(U, k) .* pdf.(Normal.(mean(y), s), l) 
contour(-1:0.1:11,3.5:0.1:6, (k,l)->plt(k,l), xlabel=L"\bar{x}", ylabel=L"y", levels=15)



