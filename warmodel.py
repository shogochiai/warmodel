import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import os

class EnhancedMultiNationalEconomicWarRiskModel:
    def __init__(self, num_countries, num_clusters, initial_conditions, disaster_scenario, aging_scenario, automation_level):
        self.num_countries = num_countries
        self.num_clusters = num_clusters
        self.initial_conditions = initial_conditions
        self.disaster_scenario = disaster_scenario
        self.aging_scenario = aging_scenario
        self.automation_level = automation_level
        
    def initialize_countries(self):
        countries = []
        base_wealth = self.initial_conditions['base_national_wealth']
        
        country_properties = {
            'United States': {'population': 331e6, 'urbanization': 0.83, 'education_level': 0.9, 'unrest_adjustment': 0.5},
            'China': {'population': 1439e6, 'urbanization': 0.61, 'education_level': 0.75, 'unrest_adjustment': 2.0},
            'Japan': {'population': 126e6, 'urbanization': 0.92, 'education_level': 0.95, 'unrest_adjustment': 0.3},
            'Germany': {'population': 83e6, 'urbanization': 0.77, 'education_level': 0.94, 'unrest_adjustment': 0.4},
            'United Kingdom': {'population': 67e6, 'urbanization': 0.84, 'education_level': 0.93, 'unrest_adjustment': 0.5},
            'France': {'population': 65e6, 'urbanization': 0.81, 'education_level': 0.92, 'unrest_adjustment': 0.7},
            'India': {'population': 1380e6, 'urbanization': 0.35, 'education_level': 0.65, 'unrest_adjustment': 1.5},
            'Italy': {'population': 60e6, 'urbanization': 0.71, 'education_level': 0.88, 'unrest_adjustment': 0.6},
            'Brazil': {'population': 212e6, 'urbanization': 0.87, 'education_level': 0.75, 'unrest_adjustment': 1.2},
            'Canada': {'population': 38e6, 'urbanization': 0.81, 'education_level': 0.93, 'unrest_adjustment': 0.3},
            'Russia': {'population': 145e6, 'urbanization': 0.75, 'education_level': 0.82, 'unrest_adjustment': 2.5},
            'South Korea': {'population': 51e6, 'urbanization': 0.82, 'education_level': 0.95, 'unrest_adjustment': 0.8},
            'Australia': {'population': 25e6, 'urbanization': 0.86, 'education_level': 0.92, 'unrest_adjustment': 0.3},
            'Spain': {'population': 47e6, 'urbanization': 0.80, 'education_level': 0.89, 'unrest_adjustment': 0.6},
            'Mexico': {'population': 128e6, 'urbanization': 0.80, 'education_level': 0.75, 'unrest_adjustment': 1.5},
            'Indonesia': {'population': 273e6, 'urbanization': 0.56, 'education_level': 0.70, 'unrest_adjustment': 1.2},
            'Netherlands': {'population': 17e6, 'urbanization': 0.92, 'education_level': 0.94, 'unrest_adjustment': 0.4},
            'Saudi Arabia': {'population': 34e6, 'urbanization': 0.84, 'education_level': 0.80, 'unrest_adjustment': 1.5},
            'Turkey': {'population': 84e6, 'urbanization': 0.76, 'education_level': 0.79, 'unrest_adjustment': 1.8},
            'Switzerland': {'population': 8.6e6, 'urbanization': 0.74, 'education_level': 0.96, 'unrest_adjustment': 0.1},
            'Nigeria': {'population': 206e6, 'urbanization': 0.52, 'education_level': 0.62, 'unrest_adjustment': 2.0},
            'South Africa': {'population': 59e6, 'urbanization': 0.67, 'education_level': 0.75, 'unrest_adjustment': 1.5},
            'Egypt': {'population': 102e6, 'urbanization': 0.43, 'education_level': 0.71, 'unrest_adjustment': 1.8},
            'Pakistan': {'population': 220e6, 'urbanization': 0.37, 'education_level': 0.59, 'unrest_adjustment': 2.2},
            'Argentina': {'population': 45e6, 'urbanization': 0.92, 'education_level': 0.88, 'unrest_adjustment': 1.0},
            'Thailand': {'population': 70e6, 'urbanization': 0.51, 'education_level': 0.78, 'unrest_adjustment': 1.2},
            'Vietnam': {'population': 97e6, 'urbanization': 0.37, 'education_level': 0.80, 'unrest_adjustment': 1.0},
            'Bangladesh': {'population': 164e6, 'urbanization': 0.39, 'education_level': 0.61, 'unrest_adjustment': 2.5},
            'Poland': {'population': 38e6, 'urbanization': 0.60, 'education_level': 0.91, 'unrest_adjustment': 0.8},
            'Iran': {'population': 84e6, 'urbanization': 0.75, 'education_level': 0.85, 'unrest_adjustment': 2.8},
            'Sweden': {'population': 10e6, 'urbanization': 0.88, 'education_level': 0.94, 'unrest_adjustment': 0.3},
            'Norway': {'population': 5.4e6, 'urbanization': 0.83, 'education_level': 0.95, 'unrest_adjustment': 0.2},
            'Ukraine': {'population': 44e6, 'urbanization': 0.69, 'education_level': 0.90, 'unrest_adjustment': 3.0},
            'Belgium': {'population': 11.6e6, 'urbanization': 0.98, 'education_level': 0.93, 'unrest_adjustment': 0.5},
            'Austria': {'population': 9e6, 'urbanization': 0.59, 'education_level': 0.93, 'unrest_adjustment': 0.4},
            'Ireland': {'population': 4.9e6, 'urbanization': 0.63, 'education_level': 0.94, 'unrest_adjustment': 0.3},
            'Israel': {'population': 9.2e6, 'urbanization': 0.92, 'education_level': 0.92, 'unrest_adjustment': 2.8},
            'Singapore': {'population': 5.8e6, 'urbanization': 1.00, 'education_level': 0.96, 'unrest_adjustment': 0.2},
            'Malaysia': {'population': 32e6, 'urbanization': 0.77, 'education_level': 0.82, 'unrest_adjustment': 1.0},
            'Philippines': {'population': 109e6, 'urbanization': 0.47, 'education_level': 0.75, 'unrest_adjustment': 1.5},
            'Colombia': {'population': 50e6, 'urbanization': 0.81, 'education_level': 0.79, 'unrest_adjustment': 1.8},
            'Chile': {'population': 19e6, 'urbanization': 0.88, 'education_level': 0.85, 'unrest_adjustment': 1.0},
            'Denmark': {'population': 5.8e6, 'urbanization': 0.88, 'education_level': 0.95, 'unrest_adjustment': 0.2},
            'Finland': {'population': 5.5e6, 'urbanization': 0.85, 'education_level': 0.94, 'unrest_adjustment': 0.3},
            'Greece': {'population': 10.4e6, 'urbanization': 0.79, 'education_level': 0.91, 'unrest_adjustment': 0.8},
            'Portugal': {'population': 10.2e6, 'urbanization': 0.66, 'education_level': 0.88, 'unrest_adjustment': 0.5},
            'New Zealand': {'population': 5e6, 'urbanization': 0.87, 'education_level': 0.93, 'unrest_adjustment': 0.2},
            'Czech Republic': {'population': 10.7e6, 'urbanization': 0.74, 'education_level': 0.93, 'unrest_adjustment': 0.5},
            'Romania': {'population': 19.2e6, 'urbanization': 0.55, 'education_level': 0.86, 'unrest_adjustment': 0.8},
            'Peru': {'population': 33e6, 'urbanization': 0.78, 'education_level': 0.81, 'unrest_adjustment': 1.2}
        }
        for _ in range(self.num_countries):
            nation = np.random.choice(list(country_properties.keys()))
            props = country_properties[nation]
            
            variation = np.random.uniform(-0.3, 0.3)
            initial_wealth = max(base_wealth * (1 + variation), 0)
            
            population = props['population']
            clusters = []
            for _ in range(self.num_clusters):
                cluster = self.initial_conditions['cluster'].copy()
                cluster['population'] = population / self.num_clusters * np.random.uniform(0.5, 2.0)
                clusters.append(cluster)
            
            countries.append({
                'wealth': initial_wealth,
                'clusters': clusters,
                'aging_coef': np.random.uniform(0.3, 1.3),
                'active': True,
                'urbanization': props['urbanization'],
                'aging_index': 20,
                'population': population,
                'gini_coefficient': np.random.uniform(0.2, 0.5),
                'productivity': 1.0,
                'nation': nation,
                'education_level': props['education_level'],
                'unrest_adjustment': props['unrest_adjustment'],
            })
        return countries
    
    def update_demographics(self, country, t, T):
        if self.aging_scenario == 'moderate':
            country['aging_index'] = min(100, 20 + 80 * (t / T) * country['aging_coef'])
        else:  # rapid aging
            country['aging_index'] = min(200, 20 + 180 * (t / T) * country['aging_coef'])
        country['urbanization'] = min(1, 0.3 + 0.7 * (t / T))
        return country
    
    def cluster_market_oscillation(self, t, cluster, urbanization, aging_index):
        A0, alpha, omega, phi = cluster['params']
        population = cluster['population']
        aging_factor = 1 - (aging_index - 50) / 200
        t_mod = t % 60
        oscillation = max(0, A0 * (np.cos(omega * t_mod + phi) + 1) * aging_factor + np.random.normal(0, 0.1))
        return oscillation * population
    
    def calculate_gini_coefficient(self, cluster_assets):
        if len(cluster_assets) == 0 or np.sum(cluster_assets) == 0:
            return 0
        sorted_assets = np.sort(cluster_assets)
        index = np.arange(1, len(cluster_assets) + 1)
        return max(0, (np.sum((2 * index - len(cluster_assets) - 1) * sorted_assets)) / (len(cluster_assets) * np.sum(sorted_assets)))
    
    def calculate_national_wealth(self, cluster_assets, tax_rate, aging_index):
        if aging_index <= 20:
            adjusted_tax_rate = tax_rate * 0.6
        elif aging_index >= 200:
            adjusted_tax_rate = tax_rate * 1.3
        else:
            adjusted_tax_rate = tax_rate * (0.6 + 0.7 * (aging_index - 20) / 180)
        return np.sum(cluster_assets) * adjusted_tax_rate
    
    def calculate_unrest_score(self, country):
        urban_rural_conflict = abs(country['urbanization'] - 0.5)
        education_resentment = country['education_level'] * (1 - country['education_level'])  # Highest at 0.5, lowest at 0 and 1
        elite_surplus = (country['education_level'] - country['wealth'] / country['population']) * 0.1
        gini_score = country['gini_coefficient'] / (country['wealth'] / country['population'])
        
        # Normalize values to 0-1 range
        urban_rural_conflict = max(min(urban_rural_conflict, 1), 0)
        education_resentment = max(min(education_resentment, 1), 0)
        elite_surplus = max(min(elite_surplus, 1), 0)
        gini_score = max(min(gini_score, 1), 0);

        return 1.0 * urban_rural_conflict + 1.0 * education_resentment + 0.3 * elite_surplus + 1.8 * gini_score
    
    
    def check_civil_unrest(self, country):
        unrest_score = self.calculate_unrest_score(country)
        unrest_probability = max(min(unrest_score * country['unrest_adjustment'], 1), 0)
        print(unrest_probability)
        return random.random() < unrest_probability
    
    def apply_disaster_penalty(self, wealth):
        if np.random.random() < self.disaster_scenario['probability']:
            impact = np.random.uniform(self.disaster_scenario['min_impact'], self.disaster_scenario['max_impact'])
            return max(0, wealth * (1 - impact)), True
        return wealth, False
    
    def apply_automation_effects(self, country):
        if self.automation_level == 'low':
            productivity_boost = 1.1
            inequality_factor = 0.99
            education_factor = 1.01
        elif self.automation_level == 'medium':
            productivity_boost = 1.3
            inequality_factor = 1.01
            education_factor = 1.03
        else:  # high
            productivity_boost = 1.5
            inequality_factor = 1.04
            education_factor = 1.05
        
        country['productivity'] *= productivity_boost
        country['gini_coefficient'] = min(1, country['gini_coefficient'] * inequality_factor)
        country['education_level'] *= education_factor
        return country
    
    def calculate_global_war_probability(self, national_wealth, urbanization_rates, aging_indices, education_levels):
        active_countries = [i for i, w in enumerate(national_wealth) if w > 0]
        if len(active_countries) < 2:
            return 0.0

        total_wealth = np.sum(national_wealth)
        avg_wealth = total_wealth / len(active_countries)
        
        # 1. 貧しい国家の比率
        poor_country_ratio = np.sum(national_wealth < 0.5 * avg_wealth) / len(active_countries)
        
        # 2. 全世界の富の大きさ
        poor_nations = [w for w in national_wealth if w < 0.5 * avg_wealth]
        wealth_gap = sum(0.5 * avg_wealth - w for w in poor_nations)
        # Prevent overflow by clamping the exponent
        exponent = np.clip(0.001 * (total_wealth - wealth_gap), -709, 709)
        global_wealth_factor = 1 / (1 + np.exp(exponent))
        
        # 戦争リスク計算
        war_risk = (
            poor_country_ratio
        ) * global_wealth_factor
        
        return min(max(war_risk, 0), 1)
    
    def apply_war_penalties(self, countries):
        active_countries = [country for country in countries if country['active']]
        if len(active_countries) < 2:
            return countries

        winner = max(active_countries, key=lambda x: x['wealth'])
        losers = [country for country in active_countries if country != winner]

        winner['wealth'] *= 0.8
        for cluster in winner['clusters']:
            cluster['params'][0] *= 0.8

        for loser in losers:
            loser['wealth'] *= 0.4
            for cluster in loser['clusters']:
                cluster['params'][0] *= 0.4

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
                country = self.apply_automation_effects(country)
                
                cluster_assets = np.array([self.cluster_market_oscillation(current_time, cluster, country['urbanization'], country['aging_index']) 
                                           for cluster in country['clusters']])
                
                gini_coefficients[i, t] = self.calculate_gini_coefficient(cluster_assets)
                new_wealth = self.calculate_national_wealth(cluster_assets, self.initial_conditions['tax_rate'], country['aging_index'])
                
                country['wealth'] = country['wealth'] + new_wealth
                country['wealth'], disaster_events[i, t] = self.apply_disaster_penalty(country['wealth'])
                
                unrest_events[i, t] = self.check_civil_unrest(country)
                if unrest_events[i, t]:
                    country['wealth'] *= 0.9  # 内乱による経済的損失
                
                national_wealth[i, t] = country['wealth']
                urbanization_rates[i, t] = country['urbanization']
                aging_indices[i, t] = country['aging_index']
                
                if country['wealth'] <= 0:
                    country['active'] = False
            
            global_war_probability[t] = self.calculate_global_war_probability(
                [country['wealth'] for country in active_countries],
                [country['urbanization'] for country in active_countries],
                [country['aging_index'] for country in active_countries],
                [country['education_level'] for country in active_countries]
            )
            if np.random.random() < global_war_probability[t]:
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

def run_simulations():
    initial_conditions = {
        'base_national_wealth': 1000,
        'cluster': {'params': [1000, 0.1, 1, 0], 'population': 1000000},
        'tax_rate': 0.3
    }

    disaster_scenarios = {
        'average': {'probability': 0.03, 'min_impact': 0.01, 'max_impact': 0.1},
        'extreme': {'probability': 0.07, 'min_impact': 0.05, 'max_impact': 0.3}
    }

    aging_scenarios = ['moderate', 'rapid']
    automation_levels = ['low', 'medium', 'high']

    results = {}

    for disaster_scenario in disaster_scenarios:
        for aging_scenario in aging_scenarios:
            for automation_level in automation_levels:
                scenario_name = f"{disaster_scenario}_{aging_scenario}_{automation_level}"
                model = EnhancedMultiNationalEconomicWarRiskModel(
                    num_countries=5,
                    num_clusters=10,
                    initial_conditions=initial_conditions,
                    disaster_scenario=disaster_scenarios[disaster_scenario],
                    aging_scenario=aging_scenario,
                    automation_level=automation_level
                )
                results[scenario_name] = model.simulate(T=1000, dt=1)

    return results


def plot_results(results):
    # フィギュア保存用のディレクトリを作成
    os.makedirs('./figs', exist_ok=True)

    # 戦争確率の時系列比較
    plt.figure(figsize=(12, 6))
    for scenario_name, result in results.items():
        plt.plot(result['global_war_probability'], label=scenario_name)
    plt.title('Global War Probability Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Probability')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('./figs/war_probability_comparison.png')
    plt.close()

    # 平均国家富の比較
    plt.figure(figsize=(12, 6))
    for scenario_name, result in results.items():
        avg_wealth = np.mean(result['national_wealth'], axis=0)
        plt.plot(avg_wealth, label=scenario_name)
    plt.title('Average National Wealth Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Wealth')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('./figs/national_wealth_comparison.png')
    plt.close()

    # 戦争発生数の時系列比較
    plt.figure(figsize=(12, 6))
    for scenario_name, result in results.items():
        war_events = np.cumsum(result['global_war_events'])
        plt.plot(war_events, label=scenario_name)
    plt.title('Cumulative Global War Events Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of War Events')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('./figs/cumulative_global_war_events.png')
    plt.close()

    # 累積災害イベント数の比較
    plt.figure(figsize=(12, 6))
    for scenario_name, result in results.items():
        cumulative_disasters = np.sum(result['disaster_events'], axis=0).cumsum()
        plt.plot(cumulative_disasters, label=scenario_name)
    plt.title('Cumulative Disaster Events')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Events')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('./figs/cumulative_disaster_events.png')
    plt.close()

    # 累積内乱イベント数の比較
    plt.figure(figsize=(12, 6))
    for scenario_name, result in results.items():
        cumulative_unrest = np.sum(result['unrest_events'], axis=0).cumsum()
        plt.plot(cumulative_unrest, label=scenario_name)
    plt.title('Cumulative Civil Unrest Events')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Events')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('./figs/cumulative_unrest_events.png')
    plt.close()

    # 平均ジニ係数の比較
    plt.figure(figsize=(12, 6))
    for scenario_name, result in results.items():
        avg_gini = np.mean(result['gini_coefficients'], axis=0)
        plt.plot(avg_gini, label=scenario_name)
    plt.title('Average Gini Coefficient Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Gini Coefficient')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('./figs/gini_coefficient_comparison.png')
    plt.close()

    # 高齢化指数の比較
    plt.figure(figsize=(12, 6))
    for scenario_name, result in results.items():
        avg_aging = np.mean(result['aging_indices'], axis=0)
        plt.plot(avg_aging, label=scenario_name)
    plt.title('Average Aging Index Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Aging Index')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('./figs/aging_index_comparison.png')
    plt.close()



    print("All plots have been saved in the './figs' directory.")

if __name__ == "__main__":
    simulation_results = run_simulations()
    plot_results(simulation_results)