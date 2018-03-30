

function one_of_K_mapping(y)
    y_values = unique(y)
    Y = zeros(length(y),length(y_values))
    for i in 1:length(y)
        for j in 1:length(y_values)
            if y[i]==y_values[j]
                Y[i,j] = 1;
                break;
            end
        end
    end
    return Y,y_values
end
