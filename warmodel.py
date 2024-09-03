import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class EnhancedMultiNationalEconomicWarRiskModel:
    def __init__(self, num_countries, num_clusters, initial_conditions, disaster_scenario, consider_urbanization=True):
        self.num_countries = num_countries
        self.num_clusters = num_clusters
        self.initial_conditions = initial_conditions
        self.disaster_scenario = disaster_scenario
        self.consider_urbanization = consider_urbanization
        
    def initialize_countries(self):
        countries = []
        base_wealth = self.initial_conditions['base_national_wealth']
        for _ in range(self.num_countries):
            variation = np.random.uniform(-0.3, 0.3)
            initial_wealth = max(base_wealth * (1 + variation), 0)
            clusters = []
            for _ in range(self.num_clusters):
                cluster = self.initial_conditions['cluster'].copy()
                cluster['population'] *= np.random.uniform(0.5, 2.0)
                clusters.append(cluster)
            countries.append({
                'wealth': initial_wealth,
                'clusters': clusters,
                'aging_coef':  np.random.uniform(0.3, 1.3), # smaller is aging-resistant society
                'active': True,
                'urbanization': 0.3,  # Initial urbanization rate
                'aging_index': 20  # Initial aging index (elderly per 100 youth)
            })
        return countries
    
    def update_demographics(self, country, t, T):
        # Simulate urbanization and aging trends to reach max at 1000 steps
        if self.consider_urbanization:
            country['urbanization'] = min(1, 0.3 + 0.7 * (t / T))
        country['aging_index'] = min(200, 20 + 180 * (t / T) * country['aging_coef'])
        return country
    
    def cluster_market_oscillation(self, t, cluster, urbanization, aging_index):
        A0, alpha, omega, phi = cluster['params']
        population = cluster['population']
        
        # Adjust oscillation based on demographics
        aging_factor = 1 - (aging_index - 50) / 200
        
        t_mod = t % 60  # Reset every 60 steps = death of a person
        oscillation = max(0, A0 * (np.cos(omega * t_mod + phi) + 1)  * aging_factor + np.random.normal(0, 0.1))
        return oscillation * population
    
    def calculate_gini_coefficient(self, cluster_assets):
        if len(cluster_assets) == 0 or np.sum(cluster_assets) == 0:
            return 0
        sorted_assets = np.sort(cluster_assets)
        index = np.arange(1, len(cluster_assets) + 1)
        return max(0, (np.sum((2 * index - len(cluster_assets) - 1) * sorted_assets)) / (len(cluster_assets) * np.sum(sorted_assets)))
    
    def calculate_national_wealth(self, cluster_assets, tax_rate, aging_index):
        # Adjust tax rate based on aging population
        # Adjust tax rate based on aging population
        if aging_index <= 20:
            adjusted_tax_rate = tax_rate * 0.6  # Very low effective tax burden for young societies
        elif aging_index >= 200:
            adjusted_tax_rate = tax_rate * 1.3  # High effective tax burden for aged societies
        else:
            adjusted_tax_rate = tax_rate * (0.6 + 0.7 * (aging_index - 20) / 180)  # Linear interpolation between young and aged societies
        return np.sum(cluster_assets) * adjusted_tax_rate
    
    def apply_civil_unrest_penalty(self, wealth, gini_coefficient, urbanization, threshold=0.8):
        # Adjust threshold based on urbanization if considered
        if self.consider_urbanization:
            adjusted_threshold = threshold - (urbanization - 0.5) * 0.1
        else:
            adjusted_threshold = threshold
        
        if gini_coefficient > adjusted_threshold:
            penalty = wealth * (gini_coefficient - adjusted_threshold)
            return max(0, wealth - penalty), True
        return wealth, False
    
    def apply_disaster_penalty(self, wealth):
        if np.random.random() < self.disaster_scenario['probability']:
            impact = np.random.uniform(self.disaster_scenario['min_impact'], self.disaster_scenario['max_impact'])
            return max(0, wealth * (1 - impact)), True
        return wealth, False
    
    def calculate_global_war_probability(self, national_wealth, urbanization_rates, aging_indices):
        active_wealth = [s for s in national_wealth if s > 0]
        if len(active_wealth) < 2:
            return 0.0
        total_wealth = np.sum(active_wealth)
        normalized_wealth = active_wealth / total_wealth
        wealth_inequality = np.std(normalized_wealth)
        low_wealth_countries = np.sum(normalized_wealth < 0.5 / len(active_wealth))
        
        base_probability = 0.01
        inequality_factor = wealth_inequality * 5
        low_wealth_factor = low_wealth_countries / len(active_wealth)
        
        # Adjust probability based on total wealth
        total_wealth_factor = 1 / (1 + np.exp(-0.5 * (total_wealth - 1000)))  # Sigmoid function
        
        # Adjust probability based on average urbanization and aging
        avg_urbanization = np.mean(urbanization_rates)
        avg_aging_index = np.mean(aging_indices)
        demographic_factor = 1 + (avg_urbanization - 0.5) * 0.5 - (avg_aging_index - 50) / 200 if self.consider_urbanization else 1 - (avg_aging_index - 50) / 200
        
        war_probability = (base_probability + inequality_factor + low_wealth_factor) * total_wealth_factor * demographic_factor
        return min(max(war_probability, 0), 1)
    
    def apply_war_penalties(self, countries):
        active_countries = [country for country in countries if country['active']]
        if len(active_countries) < 2:
            return countries

        # Determine winner and losers
        winner = max(active_countries, key=lambda x: x['wealth'])
        losers = [country for country in active_countries if country != winner]

        # Apply penalties
        winner['wealth'] *= 0.8
        for cluster in winner['clusters']:
            cluster['params'][0] *= 0.8  # Reduce A0

        for loser in losers:
            loser['wealth'] *= 0.4
            for cluster in loser['clusters']:
                cluster['params'][0] *= 0.4  # Reduce A0

        return countries
    
    def simulate(self, T, dt):
        time_steps = int(T / dt)
        global_war_probability = np.zeros(time_steps)
        national_wealth = np.zeros((self.num_countries, time_steps))
        gini_coefficients = np.zeros((self.num_countries, time_steps))
        urbanization_rates = np.zeros((self.num_countries, time_steps))
        aging_indices = np.zeros((self.num_countries, time_steps))
        global_war_events = np.zeros(time_steps, dtype=bool)
        disaster_events = np.zeros((self.num_countries, time_steps), dtype=bool)
        unrest_events = np.zeros((self.num_countries, time_steps), dtype=bool)
        
        countries = self.initialize_countries()
        
        for t in range(time_steps):
            current_time = t * dt
            active_countries = [country for country in countries if country['active']]
            
            if len(active_countries) < 2:
                break
            
            for i, country in enumerate(countries):
                if not country['active']:
                    continue
                
                country = self.update_demographics(country, t, T)
                
                cluster_assets = np.array([self.cluster_market_oscillation(current_time, cluster, country['urbanization'], country['aging_index']) 
                                           for cluster in country['clusters']])
                
                gini_coefficients[i, t] = self.calculate_gini_coefficient(cluster_assets)
                new_wealth = self.calculate_national_wealth(cluster_assets, self.initial_conditions['tax_rate'], country['aging_index'])
                
                # Adjust national wealth based on previous wealth and new income
                country['wealth'] = country['wealth'] + new_wealth
                country['wealth'], unrest_events[i, t] = self.apply_civil_unrest_penalty(country['wealth'], gini_coefficients[i, t], country['urbanization'])
                country['wealth'], disaster_events[i, t] = self.apply_disaster_penalty(country['wealth'])
                
                national_wealth[i, t] = country['wealth']
                urbanization_rates[i, t] = country['urbanization']
                aging_indices[i, t] = country['aging_index']
                
                if country['wealth'] <= 0:
                    country['active'] = False
            
            global_war_probability[t] = self.calculate_global_war_probability([country['wealth'] for country in active_countries],
                                                                              [country['urbanization'] for country in active_countries],
                                                                              [country['aging_index'] for country in active_countries])
            
            if global_war_probability[t] == 1:
                global_war_events[t] = True
                countries = self.apply_war_penalties(countries)
        
        return {
            'global_war_probability': global_war_probability,
            'national_wealth': national_wealth,
            'gini_coefficients': gini_coefficients,
            'urbanization_rates': urbanization_rates,
            'aging_indices': aging_indices,
            'global_war_events': global_war_events,
            'disaster_events': disaster_events,
            'unrest_events': unrest_events
        }

def plot_war_probability_comparison(results):
    plt.figure(figsize=(12, 6))
    for scenario_name, result in results.items():
        plt.plot(result['global_war_probability'], label=scenario_name)
    plt.title('Global War Probability Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig('./figs/war_probability_comparison.png')
    plt.close()

def plot_national_wealth_comparison(results):
    plt.figure(figsize=(12, 6))
    for scenario_name, result in results.items():
        avg_wealth = np.mean(result['national_wealth'], axis=0)
        plt.plot(avg_wealth, label=scenario_name)
    plt.title('Average National Wealth Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Wealth')
    plt.legend()
    plt.grid(True)
    plt.savefig('./figs/national_wealth_comparison.png')
    plt.close()

def plot_detailed_events(results):
    num_scenarios = len(results)
    num_countries = results[list(results.keys())[0]]['national_wealth'].shape[0]
    num_steps = results[list(results.keys())[0]]['national_wealth'].shape[1]

    # Create two separate figures for disasters and unrest
    fig_disasters, ax_disasters = plt.subplots(figsize=(14, 6))
    fig_unrest, ax_unrest = plt.subplots(figsize=(14, 6))

    fig_disasters.suptitle('Cumulative Disaster Events by Scenario and Country', fontsize=12)
    fig_unrest.suptitle('Cumulative Unrest Events by Scenario and Country', fontsize=12)

    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, num_scenarios))
    line_styles = ['-', '--', '-.', ':']

    for i, (scenario_name, result) in enumerate(results.items()):
        for country in range(num_countries):
            cumulative_disasters = np.cumsum(result['disaster_events'][country])
            cumulative_unrest = np.cumsum(result['unrest_events'][country])
            
            color = colors[i]
            line_style = line_styles[country % len(line_styles)]
            
            ax_disasters.plot(cumulative_disasters, label=f'{scenario_name} - Country {country+1}', 
                              linestyle=line_style, color=color)
            ax_unrest.plot(cumulative_unrest, label=f'{scenario_name} - Country {country+1}', 
                           linestyle=line_style, color=color)

    for ax, event_type in [(ax_disasters, 'Disasters'), (ax_unrest, 'Unrest')]:
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Cumulative Event Count')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True)

        # Add a text box explaining the line styles
        line_style_text = '\n'.join([f"Country {i+1}: {style}" for i, style in enumerate(line_styles)])
        ax.text(1.05, 0.5, f"Line Styles:\n{line_style_text}", transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8), verticalalignment='bottom')

        # Adjust the plot area to make room for legends
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    fig_disasters.savefig('./figs/cumulative_disaster_events.png')
    fig_unrest.savefig('./figs/cumulative_unrest_events.png')
    plt.close(fig_disasters)
    plt.close(fig_unrest)

# Run simulations for all scenarios
initial_conditions = {
    'base_national_wealth': 1000,
    'cluster': {'params': [1000, 0.1, 1, 0], 'population': 1000000 },
    'tax_rate': 0.3
}

disaster_scenarios = {
    'low': {'probability': 0.01, 'min_impact': 0.01, 'max_impact': 0.05},
    'high': {'probability': 0.07, 'min_impact': 0.05, 'max_impact': 0.3},
    'veryhigh': {'probability': 0.15, 'min_impact': 0.1, 'max_impact': 0.6}
}

scenarios = [
    ('Low dis., urbanized', disaster_scenarios['low'], True),
    ('Low dis., anti-urban', disaster_scenarios['low'], False),
    ('High dis., urbanized', disaster_scenarios['high'], True),
    ('High dis., anti-urban', disaster_scenarios['high'], False),
    ('v-High dis., urbanized', disaster_scenarios['veryhigh'], True),
    ('v-High dis., anti-urban', disaster_scenarios['veryhigh'], False)
]

results = {}

for scenario_name, disaster_scenario, consider_urbanization in scenarios:
    model = EnhancedMultiNationalEconomicWarRiskModel(
        num_countries=5, 
        num_clusters=10, 
        initial_conditions=initial_conditions, 
        disaster_scenario=disaster_scenario,
        consider_urbanization=consider_urbanization
    )
    results[scenario_name] = model.simulate(T=1000, dt=1)

# Ensure the ./figs directory exists
# Generate all plots
plot_war_probability_comparison(results)
plot_national_wealth_comparison(results)
plot_detailed_events(results)

# Calculate and print summary statistics
for scenario_name, result in results.items():
    print(f"\nScenario: {scenario_name}")
    print(f"Final war probability: {result['global_war_probability'][-1]:.4f}")
    print(f"Average war probability: {np.mean(result['global_war_probability']):.4f}")
    print(f"Total wars: {np.sum(result['global_war_events'])}")
    print(f"Final average wealth: {np.mean(result['national_wealth'][:, -1]):.2f}")
    print(f"Overall average wealth: {np.mean(result['national_wealth']):.2f}")
    print("Aging indices for each country:")
    for i, country_aging in enumerate(result['aging_indices']):
        print(f"  Country {i+1}: Initial: {country_aging[0]:.2f}, Final: {country_aging[-1]:.2f}")