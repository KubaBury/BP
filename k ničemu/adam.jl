using Flux
using Plots

# generate data
truef(x) = 5*ℯ.^(-1/2*((x.+0.5).^2/9))
xfine = -5.5:0.01:4.5; # fine grid for display
plot(xfine, truef) # plot  function on a fine grid
xv = -5:1:5;      # grid for data generation
y = truef(xv) .+ 0.1randn(size(xv));

# prepare model of the data
model(x,θ) =  θ[1]ℯ.^(-1/2*((x.+θ[2]).^2/((θ[3]).^2)))
# prepare initial value of the parameters
θe = [1.,1.,1.];
# minimize square error of the model
loss(x,y)= sum((y.-model(x,θe)).^2);

# prepare data iterator that 100 times repeat pairs (x,y) 
data = Iterators.repeated((xv,y),100)
# find best fit of the parameters
Flux.train!(loss,Flux.params(θe),data,ADAM(0.1))

# show data
scatter!(xv,y)
plot!(xfine,x->model(x,θe))


# see how it works => split training into 10 epochs and show progress
θe = [1.,1.,1.];
TH=zeros(3,10);
for epo=1:10
    global TH
    Flux.train!(loss,Flux.params(θe),data,ADAM(0.1))
    TH[:,epo].=θe;
end
TH
