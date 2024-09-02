import numpy as np
import matplotlib.pyplot as plt
import json

class EnhancedMultiNationalEconomicWarRiskModel:
    def __init__(self, num_countries, num_clusters, initial_conditions, disaster_scenarios):
        self.num_countries = num_countries
        self.num_clusters = num_clusters
        self.initial_conditions = initial_conditions
        self.disaster_scenarios = disaster_scenarios
        
    def initialize_countries(self):
        countries = []
        base_savings = self.initial_conditions['base_national_savings']
        for _ in range(self.num_countries):
            variation = np.random.uniform(-0.3, 0.3)
            initial_savings = max(base_savings * (1 + variation), 0)
            clusters = [self.initial_conditions['cluster'].copy() for _ in range(self.num_clusters)]
            countries.append({
                'savings': initial_savings,
                'clusters': clusters,
                'active': True
            })
        return countries
    
    def cluster_market_oscillation(self, t, cluster):
        A0, alpha, omega, phi = cluster['params']
        population = cluster['population']
        oscillation = max(0, A0 * np.exp(-alpha * t) * np.cos(omega * t + phi) + np.random.normal(0, 0.1))
        return oscillation * population
    
    def calculate_gini_coefficient(self, cluster_assets):
        if len(cluster_assets) == 0 or np.sum(cluster_assets) == 0:
            return 0
        sorted_assets = np.sort(cluster_assets)
        index = np.arange(1, len(cluster_assets) + 1)
        return max(0, (np.sum((2 * index - len(cluster_assets) - 1) * sorted_assets)) / (len(cluster_assets) * np.sum(sorted_assets)))
    
    def calculate_national_savings(self, cluster_assets, tax_rate):
        return np.sum(cluster_assets) * tax_rate
    
    def apply_civil_unrest_penalty(self, savings, gini_coefficient, threshold=0.8):
        if gini_coefficient > threshold:
            penalty = savings * (gini_coefficient - threshold)
            return max(0, savings - penalty), True
        return savings, False
    
    def apply_disaster_penalty(self, savings, scenario):
        if np.random.random() < scenario['probability']:
            impact = np.random.uniform(scenario['min_impact'], scenario['max_impact'])
            return max(0, savings * (1 - impact)), True
        return savings, False
    
    def calculate_global_war_probability(self, national_savings):
        active_savings = [s for s in national_savings if s > 0]
        if len(active_savings) < 2:
            return 0.0
        total_savings = np.sum(active_savings)
        normalized_savings = active_savings / total_savings
        savings_inequality = np.std(normalized_savings)
        low_savings_countries = np.sum(normalized_savings < 0.5 / len(active_savings))
        
        base_probability = 0.01
        inequality_factor = savings_inequality * 5
        low_savings_factor = low_savings_countries / len(active_savings)
        
        # Adjust probability based on total savings
        total_savings_factor = 1 / (1 + np.exp(-0.5 * (total_savings - 1000)))  # Sigmoid function
        
        war_probability = (base_probability + inequality_factor + low_savings_factor) * total_savings_factor
        return min(max(war_probability, 0), 1)
    
    def apply_war_penalties(self, countries):
        active_countries = [country for country in countries if country['active']]
        if len(active_countries) < 2:
            return countries

        # Determine winner and losers
        winner = max(active_countries, key=lambda x: x['savings'])
        losers = [country for country in active_countries if country != winner]

        # Apply penalties
        winner['savings'] *= 0.8
        for cluster in winner['clusters']:
            cluster['params'][0] *= 0.8  # Reduce A0

        for loser in losers:
            loser['savings'] *= 0.4
            for cluster in loser['clusters']:
                cluster['params'][0] *= 0.4  # Reduce A0

        return countries
    
    def apply_innovation_boost(self, countries):
        for country in countries:
            if country['active']:
                for cluster in country['clusters']:
                    cluster['params'][0] *= 2  # Double A0
        return countries
    
    def simulate(self, T, dt, disaster_scenario):
        time_steps = int(T / dt)
        global_war_probability = np.zeros(time_steps)
        national_savings = np.zeros((self.num_countries, time_steps))
        gini_coefficients = np.zeros((self.num_countries, time_steps))
        civil_unrest_events = np.zeros((self.num_countries, time_steps), dtype=bool)
        disaster_events = np.zeros((self.num_countries, time_steps), dtype=bool)
        global_war_events = np.zeros(time_steps, dtype=bool)
        country_extinctions = np.zeros((self.num_countries, time_steps), dtype=bool)
        innovation_events = np.zeros(time_steps, dtype=bool)
        
        countries = self.initialize_countries()
        
        for t in range(time_steps):
            current_time = t * dt
            active_countries = [country for country in countries if country['active']]
            
            if len(active_countries) < 2:
                break
            
            for i, country in enumerate(countries):
                if not country['active']:
                    continue
                
                cluster_assets = np.array([self.cluster_market_oscillation(current_time, cluster) 
                                           for cluster in country['clusters']])
                
                gini_coefficients[i, t] = self.calculate_gini_coefficient(cluster_assets)
                new_savings = self.calculate_national_savings(cluster_assets, self.initial_conditions['tax_rate'])
                
                # Adjust national savings based on previous savings and new income
                country['savings'] = country['savings'] + new_savings
                country['savings'], civil_unrest_events[i, t] = self.apply_civil_unrest_penalty(country['savings'], gini_coefficients[i, t])
                country['savings'], disaster_events[i, t] = self.apply_disaster_penalty(country['savings'], disaster_scenario)
                
                national_savings[i, t] = country['savings']
                
                if country['savings'] <= 0:
                    country['active'] = False
                    country_extinctions[i, t] = True
            
            global_war_probability[t] = self.calculate_global_war_probability([country['savings'] for country in active_countries])
            
            if global_war_probability[t] == 1:
                global_war_events[t] = True
                countries = self.apply_war_penalties(countries)
                countries = self.apply_innovation_boost(countries)
                innovation_events[t] = True
        
        return {
            'global_war_probability': global_war_probability,
            'national_savings': national_savings,
            'gini_coefficients': gini_coefficients,
            'civil_unrest_events': civil_unrest_events,
            'disaster_events': disaster_events,
            'global_war_events': global_war_events,
            'country_extinctions': country_extinctions,
            'innovation_events': innovation_events
        }

    def plot_results(self, results, scenario_name):
        time_steps = len(results['global_war_probability'])
        time = np.arange(time_steps)

        # Plot global war probability, events, innovations, and extinctions
        plt.figure(figsize=(12, 6))
        plt.plot(time, results['global_war_probability'], label='War Probability')
        plt.scatter(time[results['global_war_events']], 
                    results['global_war_probability'][results['global_war_events']], 
                    color='red', label='War Event', zorder=5)
        plt.scatter(time[results['innovation_events']], 
                    [0.5] * np.sum(results['innovation_events']), 
                    color='green', marker='^', label='Innovation Event', zorder=5)
        
        # Add extinction events
        total_extinction = np.all(results['country_extinctions'], axis=0)
        if np.any(total_extinction):
            extinction_time = time[total_extinction][0]
            plt.axvline(x=extinction_time, color='black', linestyle='--', linewidth=2, label='Total Extinction')
        
        plt.title(f'Global War Probability, Events, Innovations, and Extinctions Over Time ({scenario_name})')
        plt.xlabel('Time Steps')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot national savings and country extinctions
        plt.figure(figsize=(12, 6))
        for i in range(self.num_countries):
            plt.plot(time, results['national_savings'][i], label=f'Country {i+1}')
            extinction_times = time[results['country_extinctions'][i]]
            plt.scatter(extinction_times, [0] * len(extinction_times), color='black', marker='x', s=100, label='Extinction' if i == 0 else "")
        
        # Highlight total extinction
        if np.any(total_extinction):
            plt.axvline(x=extinction_time, color='red', linestyle='--', linewidth=2, label='Total Extinction')
        
        plt.title(f'National Savings Over Time ({scenario_name})')
        plt.xlabel('Time Steps')
        plt.ylabel('Savings')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot Gini coefficients
        plt.figure(figsize=(12, 6))
        for i in range(self.num_countries):
            plt.plot(time, results['gini_coefficients'][i], label=f'Country {i+1}')
        plt.title(f'Gini Coefficients Over Time ({scenario_name})')
        plt.xlabel('Time Steps')
        plt.ylabel('Gini Coefficient')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot civil unrest and disaster events
        plt.figure(figsize=(12, 6))
        for i in range(self.num_countries):
            country_extinction_time = time[results['country_extinctions'][i]][0] if np.any(results['country_extinctions'][i]) else time[-1]
            
            civil_unrest_mask = results['civil_unrest_events'][i] & (time < country_extinction_time)
            plt.scatter(time[civil_unrest_mask], [i+0.1] * np.sum(civil_unrest_mask), 
                        label='Civil Unrest' if i == 0 else "", marker='|', color='blue')
            
            disaster_mask = results['disaster_events'][i] & (time < country_extinction_time)
            plt.scatter(time[disaster_mask], [i-0.1] * np.sum(disaster_mask), 
                        label='Disaster' if i == 0 else "", marker='|', color='red')
            
            plt.scatter(country_extinction_time, i, color='black', marker='x', s=100, label='Extinction' if i == 0 else "")

        plt.title(f'Civil Unrest and Disaster Events Over Time ({scenario_name})')
        plt.xlabel('Time Steps')
        plt.ylabel('Countries')
        plt.yticks(range(self.num_countries), [f'Country {i+1}' for i in range(self.num_countries)])
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_results(self, results, filename):
        with open(filename, 'w') as f:
            json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()}, f)

# Usage example
initial_conditions = {
    'base_national_savings': 1000,
    'cluster': {'params': [1000, 0.1, 1, 0], 'population': 1000000},
    'tax_rate': 0.3
}

disaster_scenarios = {
    'low': {'probability': 0.01, 'min_impact': 0.01, 'max_impact': 0.05},
    'medium': {'probability': 0.05, 'min_impact': 0.05, 'max_impact': 0.15},
    'high': {'probability': 0.1, 'min_impact': 0.1, 'max_impact': 0.3}
}

model = EnhancedMultiNationalEconomicWarRiskModel(num_countries=5, num_clusters=10, initial_conditions=initial_conditions, disaster_scenarios=disaster_scenarios)

for scenario_name, scenario in disaster_scenarios.items():
    results = model.simulate(T=100, dt=0.1, disaster_scenario=scenario)
    model.plot_results(results, scenario_name)
    model.save_results(results, f'simulation_results_{scenario_name}.json')