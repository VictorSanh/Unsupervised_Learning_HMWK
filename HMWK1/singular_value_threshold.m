function S = singular_value_threshold(Y, tau)
    S = (Y>=tau).*(Y-tau) + (Y<=-tau).*(Y+tau);
end