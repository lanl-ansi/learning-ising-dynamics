% MATLAB script to process the data from Error Modes Paper
% and convert them to a structure that can be used for 
% learning Ising Models (T-regime)

%% File location (According to different stimuli)
n_data = 3;

switch n_data
    case 1
        % randomly moving bar
        file_loc = "randomly_moving_bar/data.mat";
    case 2
        % repeated natural movie
        file_loc = "repeated_natural_movie/data.mat";
    case 3
        % unique natural movie
        file_loc = "unique_natural_movie/data.mat";
    case 4
        % white noise checkerboard
        file_loc = "white_noise_checkerboard/data.mat";
    otherwise
        disp("Check n_data!")
end

%% Data Processing
% Read the data
S = load(file_loc);
data_spike_train = S.data;

%% Parameters for processing
% Not sure what data_spike_train.spike_times.fit_spike_times is yet
%bin_size = data_spike_train.hmm_fit.bin_size; % Picking the default value for now
bin_size = 200;
sampling_rate = data_spike_train.spike_times.sampling_rate_in_Hz;

% Set the time interval over which we are going to process the data [t0,t1]
% Note: avoiding repetitions for now
t0 = data_spike_train.stimulus.repeat_begin_times(1);
t1 = data_spike_train.stimulus.repeat_end_times(1);

% t1_binned < t1 due to it having to be an integer of bin size
t1_binned = bin_size*floor(t1/bin_size);

% Pick the 'm' value that our learning code cares about
m = floor(t1/bin_size);

% Not all the neurons are used, pick relevant ones
cells_used = data_spike_train.spike_times.cells_used;
n_cells_used = length(cells_used);

% Pick subset of cells if desired
FLAG_subset_cells = 1;

if FLAG_subset_cells
    max_cells_used = 50;
    n_cells_used = min(max_cells_used,n_cells_used);
end

FLAG_skip = 1;

if FLAG_skip
    %skip_cells = [1,7,17,21,23,31,33,35,37,38,39,40];
    skip_cells = [7,17,36,39,40,41,43,47];
    nodes_array = 1:n_cells_used;
    nodes_array(skip_cells) = [];
    n_cells_used = length(nodes_array);
else
    nodes_array = 1:n_cells_used;
end

%% Process data
% Get the spike times up to t1_binned
spike_times = cell(n_cells_used,1);

disp('Starting creation of spike_trains from data');
for ind_cell=1:n_cells_used
    node_temp = nodes_array(ind_cell);
    spike_times{ind_cell} = slice_spike_train(data_spike_train.spike_times.all_spike_times{node_temp}, t0, t1_binned);
end

% Bin the spike_times and get samples (T-regime)
disp('Binning spike_trains and creating series of spin configurations');
[samples_T_series, nodes_updated] = bin_spike_trains(spike_times, n_cells_used, m, t0, t1_binned);

% Select only those nodes where there is a flip
[samples_pairs_T, node_identities, m_eff] = select_spike_pairs(samples_T_series, nodes_updated, m, n_cells_used);

%% Plotting
% plot the spike raster image
figure(1);
set(gcf, 'Position', [0 0 1600 800]);
imshow(1-samples_T_series(:,1:250));
xlabel('Binned Times');
ylabel('Cell Number');
xlim();
ylim();
axis on;
ax = gca;
% Requires R2020a or later
exportgraphics(ax,'spike_raster_5s_natural_movie2.png','Resolution',300) 

%% Save
save_filename = sprintf('samples_T_%d_binsize_%d_nspins_%d_v2.mat', n_data, bin_size, n_cells_used);
save(save_filename, 'samples_pairs_T', 'm_eff', 'n_cells_used', 'node_identities');

function [samples_pairs_T, node_identities, m_eff] = select_spike_pairs(samples_T_series, nodes_updated, m, n_cells_used)
    % Locations in samples_T_series where there is a flip and only one node
    % being updated
    ind_flips = [];
    
    for i=1:m
        if length(nodes_updated{i}) == 1
            ind_flips = cat(1, ind_flips,i);
        end
    end
    m_eff = length(ind_flips);
    
    node_identities = zeros(m_eff,1);
    for i=1:m_eff
        node_identities(i) = nodes_updated{ind_flips(i)};
    end
    
    % n_cells_used is basically the spin_number
    samples_pairs_T = zeros(m_eff, 2*n_cells_used + 1);
    
    for i = 1:m_eff
        ind_temp = ind_flips(i);
        samples_pairs_T(i,1) = node_identities(i);
        samples_pairs_T(i,2:(n_cells_used+1)) = 2*samples_T_series(:,ind_temp)-1;
        samples_pairs_T(i,(n_cells_used+2):end) = 2*samples_T_series(:,ind_temp+1)-1;
    end
end

function [samples_T, nodes_updated] = bin_spike_trains(spike_times, n_cells_used, m, t0, t1_binned)
    % Create array of samples (row is cell, col is spin)
    samples_T = zeros(n_cells_used, m);
    
    % Create cell array of nodes being updated
    nodes_updated = cell(m,1);
    
    % Get bin edges
    % Ref: https://www.mathworks.com/help/matlab/ref/histcounts.html
    bin_edges = ceil(linspace(t0,t1_binned,m+1));
    
    for ind_cell=1:n_cells_used
        % Histogram the data into above bins
        binned_spiked_times = histcounts(spike_times{ind_cell}, bin_edges);
        
        % Get all the non-zero bin indices
        ind_nnz = find(binned_spiked_times);
        
        % Set these samples as 1
        samples_T(ind_cell,ind_nnz) = 1;
    end
    
    for ind_t=2:m
        samples_temp_diff = samples_T(:,ind_t) - samples_T(:, ind_t-1);
        nodes_updated{ind_t} = find(samples_temp_diff);
    end
end

function spike_train = slice_spike_train(spike_times, t0, t1_binned)
    % Get the index up to which to consider the spike_times
    ind = find(spike_times <= t1_binned, 1, 'last');
    spike_train = spike_times(t0:ind);
end