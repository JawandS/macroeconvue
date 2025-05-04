# Dependencies
import BeforeIT as Bit
using Plots

# Set parameters
parameters = Bit.AUSTRIA2010Q1.parameters
initial_conditions = Bit.AUSTRIA2010Q1.initial_conditions

# Set number of epochs
T = 16
model = Bit.init_model(parameters, initial_conditions, T)

# Inspect model: fieldnames(typeof(model))
# Inspect attribute: fieldnames(typeof(model.bank))

# Data tracker
data = Bit.init_data(model)

# Run simulation
for t in 1:T
    # Run one epoch
    Bit.step!(model; multi_threading = true)
    
    # Update data tracker
    Bit.update_data!(data, model)

    # Calculate change in GDP
    if t > 1
        delta_rgdp = 100 * (data.real_gdp[t] - data.real_gdp[t-1]) / data.real_gdp[t-1]
        println("Time $t: ΔRGDP = $(round(delta_rgdp; digits = 2))%")
    end

    # Print active real_household_consumption
    println("$t: $(model.w_act.C_h_sector[1])")
end

# Plot
# ps = Bit.plot_data(data, quantities = [:real_gdp, :real_household_consumption, :real_government_consumption, :real_capitalformation, :real_exports, :real_imports, :wages, :euribor, :gdp_deflator])
# plot(ps..., layout = (3, 3))
