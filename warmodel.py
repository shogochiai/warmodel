import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import os

class EnhancedMultiNationalEconomicWarRiskModel:
    def __init__(self, num_countries, num_clusters, initial_conditions, disaster_scenario, aging_scenario, automation_level):
        self.country_properties = {
            'United States': {'population': 331e6, 'urbanization': 0.83, 'education_level': 0.9, 'unrest_adjustment': 0.5, 'aging_index': 15, 'gdp': 21433225e6, 'gini': 0.41},
            'China': {'population': 1439e6, 'urbanization': 0.61, 'education_level': 0.75, 'unrest_adjustment': 2.0, 'aging_index': 30, 'gdp': 14342903e6, 'gini': 0.38},
            'Japan': {'population': 126e6, 'urbanization': 0.92, 'education_level': 0.95, 'unrest_adjustment': 0.3, 'aging_index': 35, 'gdp': 5082465e6, 'gini': 0.33},
            'Germany': {'population': 83e6, 'urbanization': 0.77, 'education_level': 0.94, 'unrest_adjustment': 0.4, 'aging_index': 25, 'gdp': 3845630e6, 'gini': 0.31},
            'United Kingdom': {'population': 67e6, 'urbanization': 0.84, 'education_level': 0.93, 'unrest_adjustment': 0.5, 'aging_index': 25, 'gdp': 2827113e6, 'gini': 0.35},
            'France': {'population': 65e6, 'urbanization': 0.81, 'education_level': 0.92, 'unrest_adjustment': 0.7, 'aging_index': 25, 'gdp': 2715518e6, 'gini': 0.32},
            'India': {'population': 1380e6, 'urbanization': 0.35, 'education_level': 0.65, 'unrest_adjustment': 1.5, 'aging_index': 18, 'gdp': 2710996e6, 'gini': 0.35},
            'Italy': {'population': 60e6, 'urbanization': 0.71, 'education_level': 0.88, 'unrest_adjustment': 0.6, 'aging_index': 28, 'gdp': 2001244e6, 'gini': 0.35},
            'Brazil': {'population': 212e6, 'urbanization': 0.87, 'education_level': 0.75, 'unrest_adjustment': 1.2, 'aging_index': 20, 'gdp': 1839758e6, 'gini': 0.53},
            'Canada': {'population': 38e6, 'urbanization': 0.81, 'education_level': 0.93, 'unrest_adjustment': 0.3, 'aging_index': 20, 'gdp': 1736425e6, 'gini': 0.33},
            'Russia': {'population': 145e6, 'urbanization': 0.75, 'education_level': 0.82, 'unrest_adjustment': 2.5, 'aging_index': 22, 'gdp': 1657554e6, 'gini': 0.37},
            'South Korea': {'population': 51e6, 'urbanization': 0.82, 'education_level': 0.95, 'unrest_adjustment': 0.8, 'aging_index': 32, 'gdp': 1630869e6, 'gini': 0.31},
            'Australia': {'population': 25e6, 'urbanization': 0.86, 'education_level': 0.92, 'unrest_adjustment': 0.3, 'aging_index': 20, 'gdp': 1396567e6, 'gini': 0.34},
            'Spain': {'population': 47e6, 'urbanization': 0.80, 'education_level': 0.89, 'unrest_adjustment': 0.6, 'aging_index': 27, 'gdp': 1394116e6, 'gini': 0.34},
            'Mexico': {'population': 128e6, 'urbanization': 0.80, 'education_level': 0.75, 'unrest_adjustment': 1.5, 'aging_index': 20, 'gdp': 1268870e6, 'gini': 0.45},
            'Indonesia': {'population': 273e6, 'urbanization': 0.56, 'education_level': 0.70, 'unrest_adjustment': 1.2, 'aging_index': 18, 'gdp': 1119190e6, 'gini': 0.38},
            'Netherlands': {'population': 17e6, 'urbanization': 0.92, 'education_level': 0.94, 'unrest_adjustment': 0.4, 'aging_index': 25, 'gdp': 909070e6, 'gini': 0.28},
            'Saudi Arabia': {'population': 34e6, 'urbanization': 0.84, 'education_level': 0.80, 'unrest_adjustment': 1.5, 'aging_index': 15, 'gdp': 792967e6, 'gini': 0.45},
            'Turkey': {'population': 84e6, 'urbanization': 0.76, 'education_level': 0.79, 'unrest_adjustment': 1.8, 'aging_index': 20, 'gdp': 761425e6, 'gini': 0.41},
            'Switzerland': {'population': 8.6e6, 'urbanization': 0.74, 'education_level': 0.96, 'unrest_adjustment': 0.1, 'aging_index': 25, 'gdp': 731502e6, 'gini': 0.33},
            'Nigeria': {'population': 206e6, 'urbanization': 0.52, 'education_level': 0.62, 'unrest_adjustment': 2.0, 'aging_index': 12, 'gdp': 448120e6, 'gini': 0.35},
            'South Africa': {'population': 59e6, 'urbanization': 0.67, 'education_level': 0.75, 'unrest_adjustment': 1.5, 'aging_index': 15, 'gdp': 351432e6, 'gini': 0.63},
            'Egypt': {'population': 102e6, 'urbanization': 0.43, 'education_level': 0.71, 'unrest_adjustment': 1.8, 'aging_index': 15, 'gdp': 303410e6, 'gini': 0.31},
            'Pakistan': {'population': 220e6, 'urbanization': 0.37, 'education_level': 0.59, 'unrest_adjustment': 2.2, 'aging_index': 15, 'gdp': 278222e6, 'gini': 0.33},
            'Argentina': {'population': 45e6, 'urbanization': 0.92, 'education_level': 0.88, 'unrest_adjustment': 1.0, 'aging_index': 20, 'gdp': 449663e6, 'gini': 0.41},
            'Thailand': {'population': 70e6, 'urbanization': 0.51, 'education_level': 0.78, 'unrest_adjustment': 1.2, 'aging_index': 22, 'gdp': 501795e6, 'gini': 0.35},
            'Vietnam': {'population': 97e6, 'urbanization': 0.37, 'education_level': 0.80, 'unrest_adjustment': 1.0, 'aging_index': 20, 'gdp': 261921e6, 'gini': 0.35},
            'Bangladesh': {'population': 164e6, 'urbanization': 0.39, 'education_level': 0.61, 'unrest_adjustment': 2.5, 'aging_index': 18, 'gdp': 302571e6, 'gini': 0.32},
            'Poland': {'population': 38e6, 'urbanization': 0.60, 'education_level': 0.91, 'unrest_adjustment': 0.8, 'aging_index': 25, 'gdp': 594165e6, 'gini': 0.30},
            'Iran': {'population': 84e6, 'urbanization': 0.75, 'education_level': 0.85, 'unrest_adjustment': 2.8, 'aging_index': 18, 'gdp': 495687e6, 'gini': 0.40},
            'Sweden': {'population': 10e6, 'urbanization': 0.88, 'education_level': 0.94, 'unrest_adjustment': 0.3, 'aging_index': 25, 'gdp': 530884e6, 'gini': 0.29},
            'Norway': {'population': 5.4e6, 'urbanization': 0.83, 'education_level': 0.95, 'unrest_adjustment': 0.2, 'aging_index': 25, 'gdp': 403336e6, 'gini': 0.27},
            'Ukraine': {'population': 44e6, 'urbanization': 0.69, 'education_level': 0.90, 'unrest_adjustment': 3.0, 'aging_index': 25, 'gdp': 155582e6, 'gini': 0.26},
            'Belgium': {'population': 11.6e6, 'urbanization': 0.98, 'education_level': 0.93, 'unrest_adjustment': 0.5, 'aging_index': 25, 'gdp': 521860e6, 'gini': 0.27},
            'Austria': {'population': 9e6, 'urbanization': 0.59, 'education_level': 0.93, 'unrest_adjustment': 0.4, 'aging_index': 25, 'gdp': 446315e6, 'gini': 0.30},
            'Ireland': {'population': 4.9e6, 'urbanization': 0.63, 'education_level': 0.94, 'unrest_adjustment': 0.3, 'aging_index': 20, 'gdp': 398476e6, 'gini': 0.32},
            'Israel': {'population': 9.2e6, 'urbanization': 0.92, 'education_level': 0.92, 'unrest_adjustment': 2.8, 'aging_index': 20, 'gdp': 402639e6, 'gini': 0.39},
            'Singapore': {'population': 5.8e6, 'urbanization': 1.00, 'education_level': 0.96, 'unrest_adjustment': 0.2, 'aging_index': 28, 'gdp': 372063e6, 'gini': 0.45},
            'Malaysia': {'population': 32e6, 'urbanization': 0.77, 'education_level': 0.82, 'unrest_adjustment': 1.0, 'aging_index': 18, 'gdp': 364681e6, 'gini': 0.41},
            'Philippines': {'population': 109e6, 'urbanization': 0.47, 'education_level': 0.75, 'unrest_adjustment': 1.5, 'aging_index': 18, 'gdp': 362243e6, 'gini': 0.44},
            'Colombia': {'population': 50e6, 'urbanization': 0.81, 'education_level': 0.79, 'unrest_adjustment': 1.8, 'aging_index': 20, 'gdp': 323803e6, 'gini': 0.51},
            'Chile': {'population': 19e6, 'urbanization': 0.88, 'education_level': 0.85, 'unrest_adjustment': 1.0, 'aging_index': 20, 'gdp': 282318e6, 'gini': 0.44},
            'Denmark': {'population': 5.8e6, 'urbanization': 0.88, 'education_level': 0.95, 'unrest_adjustment': 0.2, 'aging_index': 25, 'gdp': 355675e6, 'gini': 0.28},
            'Finland': {'population': 5.5e6, 'urbanization': 0.85, 'education_level': 0.94, 'unrest_adjustment': 0.3, 'aging_index': 25, 'gdp': 269296e6, 'gini': 0.27},
            'Greece': {'population': 10.4e6, 'urbanization': 0.79, 'education_level': 0.91, 'unrest_adjustment': 0.8, 'aging_index': 28, 'gdp': 209853e6, 'gini': 0.34},
            'Portugal': {'population': 10.2e6, 'urbanization': 0.66, 'education_level': 0.88, 'unrest_adjustment': 0.5, 'aging_index': 28, 'gdp': 231854e6, 'gini': 0.33},
            'New Zealand': {'population': 5e6, 'urbanization': 0.87, 'education_level': 0.93, 'unrest_adjustment': 0.2, 'aging_index': 20, 'gdp': 206929e6, 'gini': 0.32},
            'Czech Republic': {'population': 10.7e6, 'urbanization': 0.74, 'education_level': 0.93, 'unrest_adjustment': 0.5, 'aging_index': 25, 'gdp': 243530e6, 'gini': 0.25},
            'Romania': {'population': 19.2e6, 'urbanization': 0.55, 'education_level': 0.86, 'unrest_adjustment': 0.8, 'aging_index': 25, 'gdp': 248715e6, 'gini': 0.35},
            'Peru': {'population': 33e6, 'urbanization': 0.78, 'education_level': 0.81, 'unrest_adjustment': 1.2, 'aging_index': 20, 'gdp': 226848e6, 'gini': 0.43}
        }
        self.num_countries = len(self.country_properties.keys())
        self.num_clusters = num_clusters
        self.initial_conditions = initial_conditions
        self.disaster_scenario = disaster_scenario
        self.aging_scenario = aging_scenario
        self.automation_level = automation_level
        self.unrest_cooldown = {country: 0 for country in self.country_properties.keys()} 
        self.productivity_boost_cooldown = {country: 0 for country in self.country_properties.keys()}

    def initialize_countries(self):
        countries = []

        for country_name, props in self.country_properties.items():
            clusters = self.create_clusters(props)
            countries.append({
                'name': country_name,
                'wealth': props['gdp'],
                'clusters': clusters,
                'active': True,
                'urbanization': props['urbanization'],
                'aging_index': props['aging_index'],
                'population': props['population'],
                'education_level': props['education_level'],
                'unrest_adjustment': props['unrest_adjustment'],
                'gini_coefficient': props['gini'],
                'productivity': (props['gdp'] / props['population']) * 
                                (1 - props['gini']) * 
                                props['education_level'] * 
                                (1 + props['urbanization']) * 
                                (1 - props['aging_index'] / 100) * 
                                (1 / props['unrest_adjustment']),
                'birth_rate': self.calculate_initial_birth_rate(props),
                'death_rate': self.calculate_initial_death_rate(props),
            })
        return countries

    def calculate_initial_birth_rate(self, country_props):
        # 簡易的な出生率計算（実際のデータに基づいて調整が必要）
        return 0.02 - 0.01 * (country_props['aging_index'] / 100)  # 2% ~ 1%

    def calculate_initial_death_rate(self, country_props):
        # 簡易的な死亡率計算（実際のデータに基づいて調整が必要）
        return 0.007 + 0.003 * (country_props['aging_index'] / 100)  # 0.7% ~ 1%

    def create_clusters(self, country_props):
        clusters = []
        gini = country_props['gini']
        population = country_props['population']
        gdp = country_props['gdp']
        
        # Calculate wealth distribution based on Gini coefficient
        wealth_distribution = self.calculate_wealth_distribution(gini, population)
        
        # Create clusters and assign wealth
        num_clusters = 5  # You can adjust this number as needed
        cluster_size = population // num_clusters
        remaining_population = population % num_clusters
        
        for i in range(num_clusters):
            cluster_population = cluster_size + (1 if i < remaining_population else 0)
            cluster_wealth = (wealth_distribution[i] / 100) * gdp
            
            clusters.append({
                'population': cluster_population,
                'wealth': cluster_wealth
            })
        
        return clusters

    def calculate_wealth_distribution(self, gini, population):
        # Calculate the Lorenz curve points
        x = np.linspace(0, 1, 101)
        y = np.where(x != 1, x + (1 - x) * (1 - (1 - gini) ** (1 / np.maximum(1e-10, 1 - x))), 1)
        
        # Calculate wealth distribution for 5 quintiles
        quintiles = [20, 40, 60, 80, 100]
        wealth_distribution = []
        
        for i in range(5):
            if i == 0:
                wealth_share = y[quintiles[i]]
            else:
                wealth_share = y[quintiles[i]] - y[quintiles[i-1]]
            wealth_distribution.append(wealth_share * 100)
        
        return wealth_distribution

    def update_growth(self, country, t, T):
        if t % 10 == 0:
            # Update growth for each cluster
            for cluster in country['clusters']:
                # Apply growth rate to cluster wealth
                net_growth_rate = country['birth_rate'] - country['death_rate']
                cluster['wealth'] *= (1 + net_growth_rate)

            # Update country's total wealth
            country['wealth'] = sum(cluster['wealth'] for cluster in country['clusters'])
            return country
        else:
            return country

    def update_demographics(self, country, t, T):
        # 高齢化と都市化の更新
        if t % 10 == 0:
            if country['aging_index'] < 20:
                country['aging_index'] = min(20, country['aging_index'] + 1 * (t / T))
            else:
                if self.aging_scenario == 'moderate':
                    country['aging_index'] = min(40, country['aging_index'] + (0.3 * 70 * (t / 800) * (1 if np.random.random() < 0.6 else -1)))
                else:  # rapid aging
                    country['aging_index'] = min(140, country['aging_index'] + 80 * (t / T))
            country['urbanization'] = min(1, 0.3 + 0.7 * (t / T))

            # 人口増加の計算
            net_growth_rate = country['birth_rate'] - country['death_rate']
            for cluster in country['clusters']:
                cluster['population'] *= (1 + net_growth_rate)
            country['population'] = sum(cluster['population'] for cluster in country['clusters'])

            # 出生率と死亡率の動的な更新
            country['birth_rate'] = max(0.005, country['birth_rate'] - 0.0001 * (t / T))  # 出生率は緩やかに低下
            country['death_rate'] = min(0.02, country['death_rate'] + 0.00005 * (t / T))  # 死亡率は緩やかに上昇

            return country
        else:
            return country

    
    def calculate_gini_coefficient(self, clusters):
        total_population = sum(cluster['population'] for cluster in clusters)
        total_wealth = sum(cluster['wealth'] for cluster in clusters)
        
        # Sort clusters by wealth per capita
        sorted_clusters = sorted(clusters, key=lambda c: c['wealth'] / c['population'])
        
        cumulative_population = 0
        cumulative_wealth = 0
        area_under_lorenz = 0
        
        for cluster in sorted_clusters:
            x1 = cumulative_population / total_population
            cumulative_population += cluster['population']
            x2 = cumulative_population / total_population
            
            y1 = cumulative_wealth / total_wealth
            cumulative_wealth += cluster['wealth']
            y2 = cumulative_wealth / total_wealth
            
            area_under_lorenz += (x2 - x1) * (y1 + y2) / 2
        
        gini_coefficient = 1 - 2 * area_under_lorenz
        return gini_coefficient

        
    def calculate_national_wealth(self, country, tax_rate):
        aging_index = country['aging_index']
        
        if aging_index <= 20:
            adjusted_tax_rate = tax_rate * 0.6
        elif aging_index >= 200:
            adjusted_tax_rate = tax_rate * 1.3
        else:
            adjusted_tax_rate = tax_rate * (0.6 + 0.7 * (aging_index - 20) / 180)
        
        total_wealth = sum(cluster['wealth'] for cluster in country['clusters'])
        tax_collected = total_wealth * adjusted_tax_rate
        
        # Deduct taxes from each cluster proportionally
        for cluster in country['clusters']:
            cluster_tax = cluster['wealth'] * adjusted_tax_rate
            cluster['wealth'] -= cluster_tax
        
        # Update country's wealth
        country['wealth'] += tax_collected
        
        return country

    def calculate_unrest_score(self, country):
        urban_rural_conflict = abs(country['urbanization'] - 0.5)
        education_resentment = country['education_level'] * (1 - country['education_level'])  # Highest at 0.5, lowest at 0 and 1
        elite_surplus = (country['education_level'] - country['wealth'] / country['population']) * 0.1
        gini_score = country['gini_coefficient'] * country['population'] / country['wealth'] if country['wealth'] > 0 else 1
        
        # Normalize values to 0-1 range
        urban_rural_conflict = max(min(urban_rural_conflict, 1), 0)
        education_resentment = max(min(education_resentment, 1), 0)
        elite_surplus = max(min(elite_surplus, 1), 0)
        gini_score = max(min(gini_score, 1), 0);

        return 0.05 * urban_rural_conflict + 0.2 * education_resentment + 0.3 * elite_surplus + 1.8 * gini_score
    
    
    def check_civil_unrest(self, country):
        if self.unrest_cooldown[country['name']] > 0:
            self.unrest_cooldown[country['name']] -= 1
            return False

        unrest_score = self.calculate_unrest_score(country)
        unrest_threshold = 0.7
        unrest_probability = max(min((unrest_score - unrest_threshold) * country['unrest_adjustment'], 1), 0)
        
        if random.random() < unrest_probability:
            self.unrest_cooldown[country['name']] = 100  # 60-step cooldown period
            return True
        return False
    
    def apply_disaster_penalty(self, wealth):
        if np.random.random() < self.disaster_scenario['probability']:
            impact = np.random.uniform(self.disaster_scenario['min_impact'], self.disaster_scenario['max_impact'])
            return max(0, wealth * (1 - impact)), True
        return wealth, False
    
    def apply_automation_effects(self, country):
        if self.automation_level == 'low':
            productivity_boost = 1.1
            inequality_change_bad = (1.05, 1.10)
            inequality_change_good = (0.88, 0.94)
            education_factor = 1.01
        elif self.automation_level == 'medium':
            productivity_boost = 1.3
            inequality_change_bad = (1.07, 1.12)
            inequality_change_good = (0.86, 0.92)
            education_factor = 1.03
        else:  # high
            productivity_boost = 1.5
            inequality_change_bad = (1.09, 1.14)
            inequality_change_good = (0.84, 0.90)
            education_factor = 1.05
        
        country['productivity'] *= productivity_boost
        
        if random.random() < 0.4:  # 2 in 5 chance
            inequality_factor = random.uniform(*inequality_change_bad)
        else:  # 3 in 5 chance
            inequality_factor = random.uniform(*inequality_change_good)
        
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
        exponent = np.clip(0.001 * (total_wealth - wealth_gap), -709, 709)
        global_wealth_factor = 1 / (1 + np.exp(exponent))
        
        # 3. 新しい要因: 教育レベルの差
        education_disparity = np.std(education_levels) / np.mean(education_levels)
        
        # 4. 新しい要因: 高齢化の差
        aging_disparity = np.std(aging_indices) / np.mean(aging_indices)
        
        # 戦争リスク計算（係数を導入）
        base_risk = (
            0.1 * poor_country_ratio +
            0.2 * global_wealth_factor +
            0.15 * education_disparity +
            0.15 * aging_disparity
        )
        
        # 調整係数（この値は適切に調整する必要があります）
        adjustment_factor = 0.01
        
        war_risk = 1 - np.exp(-adjustment_factor * base_risk)
        
        return min(max(war_risk, 0), 1)


    def apply_productivity_boost(self, country):
        if self.productivity_boost_cooldown[country['name']] > 0:
            self.productivity_boost_cooldown[country['name']] -= 1
            return country, False

        if random.random() < 0.025:  # 1 in 40 chance (on average, once every 40 steps)
            boost_factor = random.uniform(1.3, 2.0)
            for cluster in country['clusters']:
                cluster['wealth'] *= boost_factor
            country['wealth'] = sum(cluster['wealth'] for cluster in country['clusters'])
            self.productivity_boost_cooldown[country['name']] = 40  # Set cooldown to 40 steps
            return country, True

        return country, False

    def apply_war_penalties(self, countries):
        active_countries = [country for country in countries if country['active']]
        if len(active_countries) < 2:
            return countries

        winner = max(active_countries, key=lambda x: x['wealth'])
        losers = [country for country in active_countries if country != winner]

        winner['wealth'] *= 0.8
        winner['birth_rate'] *= 2
        winner['death_rate'] /= 2
        winner['aging_index'] = 15  # Reset to a young population

        for loser in losers:
            loser['wealth'] *= 0.4
            loser['birth_rate'] *= 2
            loser['death_rate'] /= 2
            loser['aging_index'] = 15  # Reset to a young population

        return countries

    def redistribute_wealth_during_unrest(self, country, unrest_impact):
        total_wealth = sum(cluster['wealth'] for cluster in country['clusters'])
        total_population = sum(cluster['population'] for cluster in country['clusters'])
        average_wealth_per_capita = total_wealth / total_population

        # 勝者のクラスターを選択（貧しいクラスターがやや有利）
        clusters_with_weights = [(cluster, 1 / (cluster['wealth'] + 1)) for cluster in country['clusters']]
        total_weight = sum(weight for _, weight in clusters_with_weights)
        normalized_weights = [weight / total_weight for _, weight in clusters_with_weights]
        winner_cluster = random.choices(country['clusters'], weights=normalized_weights, k=1)[0]

        for cluster in country['clusters']:
            if cluster == winner_cluster:
                # 勝者のクラスターは富を増やす
                cluster['wealth'] *= 1.2
            else:
                # 敗者のクラスターは富を減らし、一部を勝者に移転
                reduction = cluster['wealth'] * 0.3
                cluster['wealth'] -= reduction
                winner_cluster['wealth'] += reduction * 0.5  # 減少分の半分を勝者に

        return country

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
        population_data = np.zeros((self.num_countries, time_steps))
        innovation_events = np.zeros((self.num_countries, time_steps), dtype=bool)

        countries = self.initialize_countries()
        
        for t in range(time_steps):
            current_time = t * dt
            active_countries = [country for country in countries if country['active']]
            
            if len(active_countries) < 2:
                break
            
            for i, country in enumerate(countries):
                if not country['active']:
                    continue
                


                country = self.update_growth(country, t, T)
                country = self.update_demographics(country, t, T)
                country = self.apply_automation_effects(country)
                country, innovation_occurred = self.apply_productivity_boost(country)
                if innovation_occurred:
                    innovation_events[i,t] = True

                country = self.calculate_national_wealth(country, self.initial_conditions['tax_rate'])
                country['wealth'], disaster_events[i, t] = self.apply_disaster_penalty(country['wealth'])


                unrest_events[i, t] = self.check_civil_unrest(country)
                if unrest_events[i, t]:
                    unrest_impact = np.random.uniform(0.8, 0.9)  # より小さな経済的損失
                    country['wealth'] *= unrest_impact
                    country = self.redistribute_wealth_during_unrest(country, unrest_impact)
                    country['unrest_adjustment'] *= 0.8  # 内戦後、一時的に不安定性が低下

                national_wealth[i, t] = country['wealth']
                urbanization_rates[i, t] = country['urbanization']
                aging_indices[i, t] = country['aging_index']

                gini_coefficients[i, t] = self.calculate_gini_coefficient(country['clusters'])

                population_data[i, t] = country['population'] 

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
            'unrest_events': unrest_events,
            'population_data': population_data,
            'innovation_events': innovation_events
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

    # Define meaningful labels for scenarios
    disaster_labels = {'average': 'Avg. Disasters', 'extreme': 'Extreme Disasters'}
    aging_labels = {'moderate': 'Mod. Aging', 'rapid': 'Rapid Aging'}
    automation_labels = {'low': 'Low Auto', 'medium': 'Med. Auto', 'high': 'High Auto'}

    # Create color palettes for each group
    disaster_colors = {'average': 'blue', 'extreme': 'red'}
    aging_colors = {'moderate': 'blue', 'rapid': 'red'}
    automation_colors = {'low': 'blue', 'medium': 'green', 'high': 'red'}

    # Function to create a meaningful label
    def create_label(scenario):
        disaster, aging, automation = scenario.split('_')
        return f"{disaster_labels[disaster]}, {aging_labels[aging]}, {automation_labels[automation]}"

    # Function to get line style based on scenario
    def get_line_style(disaster, aging, automation):
        if disaster == 'extreme':
            return '-.'
        elif aging == 'rapid':
            return ':'
        else:
            return '-' if automation == 'low' else '--' if automation == 'medium' else '-'

    # 戦争確率の時系列比較
    plt.figure(figsize=(15, 8))
    for scenario_name, result in results.items():
        disaster, aging, automation = scenario_name.split('_')
        color = automation_colors[automation]
        linestyle = get_line_style(disaster, aging, automation)
        plt.plot(result['global_war_probability'], label=create_label(scenario_name), color=color, linestyle=linestyle)
    plt.title('Global War Probability Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Probability')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig('./figs/war_probability_comparison.png', bbox_inches='tight')
    plt.close()

    # 平均国家富の比較
    plt.figure(figsize=(15, 8))
    for scenario_name, result in results.items():
        disaster, aging, automation = scenario_name.split('_')
        color = aging_colors[aging]
        linestyle = get_line_style(disaster, aging, automation)
        avg_wealth = np.mean(result['national_wealth'], axis=0)
        plt.plot(avg_wealth, label=create_label(scenario_name), color=color, linestyle=linestyle)
    plt.title('Average National Wealth Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Wealth')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig('./figs/national_wealth_comparison.png', bbox_inches='tight')
    plt.close()

    # 戦争発生数の時系列比較
    plt.figure(figsize=(15, 8))
    for scenario_name, result in results.items():
        disaster, aging, automation = scenario_name.split('_')
        color = automation_colors[automation]
        linestyle = get_line_style(disaster, aging, automation)
        war_events = np.cumsum(result['global_war_events'])
        plt.plot(war_events, label=create_label(scenario_name), color=color, linestyle=linestyle)
    plt.title('Cumulative Global War Events Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of War Events')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig('./figs/cumulative_global_war_events.png', bbox_inches='tight')
    plt.close()

    # 累積災害イベント数の比較
    plt.figure(figsize=(15, 8))
    for scenario_name, result in results.items():
        disaster, aging, automation = scenario_name.split('_')
        color = disaster_colors[disaster]
        linestyle = get_line_style(disaster, aging, automation)
        cumulative_disasters = np.sum(result['disaster_events'], axis=0).cumsum()
        plt.plot(cumulative_disasters, label=create_label(scenario_name), color=color, linestyle=linestyle)
    plt.title('Cumulative Disaster Events')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Events')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig('./figs/cumulative_disaster_events.png', bbox_inches='tight')
    plt.close()

    # 累積内乱イベント数の比較
    plt.figure(figsize=(15, 8))
    for scenario_name, result in results.items():
        disaster, aging, automation = scenario_name.split('_')
        color = aging_colors[aging]
        linestyle = get_line_style(disaster, aging, automation)
        cumulative_unrest = np.sum(result['unrest_events'], axis=0).cumsum()
        plt.plot(cumulative_unrest, label=create_label(scenario_name), color=color, linestyle=linestyle)
    plt.title('Cumulative Civil Unrest Events')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Events')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig('./figs/cumulative_unrest_events.png', bbox_inches='tight')
    plt.close()


    # 累積イノベーションイベント数の比較
    plt.figure(figsize=(15, 8))
    for scenario_name, result in results.items():
        disaster, aging, automation = scenario_name.split('_')
        color = automation_colors[automation]
        linestyle = get_line_style(disaster, aging, automation)
        cumulative_innovations = np.sum(result['innovation_events'], axis=0).cumsum()
        plt.plot(cumulative_innovations, label=create_label(scenario_name), color=color, linestyle=linestyle)
    plt.title('Cumulative Innovation Events')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Events')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig('./figs/cumulative_innovation_events.png', bbox_inches='tight')
    plt.close()

    # 平均ジニ係数の比較
    plt.figure(figsize=(15, 8))
    for scenario_name, result in results.items():
        disaster, aging, automation = scenario_name.split('_')
        color = automation_colors[automation]
        linestyle = get_line_style(disaster, aging, automation)
        avg_gini = np.mean(result['gini_coefficients'], axis=0)
        plt.plot(avg_gini, label=create_label(scenario_name), color=color, linestyle=linestyle)
    plt.title('Average Gini Coefficient Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Gini Coefficient')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig('./figs/gini_coefficient_comparison.png', bbox_inches='tight')
    plt.close()

    # 高齢化指数の比較
    plt.figure(figsize=(15, 8))
    for scenario_name, result in results.items():
        disaster, aging, automation = scenario_name.split('_')
        color = aging_colors[aging]
        linestyle = get_line_style(disaster, aging, automation)
        avg_aging = np.mean(result['aging_indices'], axis=0)
        plt.plot(avg_aging, label=create_label(scenario_name), color=color, linestyle=linestyle)
    plt.title('Average Aging Index Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Aging Index')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig('./figs/aging_index_comparison.png', bbox_inches='tight')
    plt.close()

    # World Population Comparison
    plt.figure(figsize=(15, 8))
    for scenario_name, result in results.items():
        disaster, aging, automation = scenario_name.split('_')
        color = automation_colors[automation]
        linestyle = get_line_style(disaster, aging, automation)
        world_population = np.sum(result['population_data'], axis=0) / 1e9  # Convert to billions
        world_population = np.clip(world_population, 0, 10)  # Clip values between 0 and 10 billion
        plt.plot(world_population, label=create_label(scenario_name), color=color, linestyle=linestyle)
    plt.title('World Population Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Population (billions)')
    plt.ylim(0, 10)  # Set y-axis limits from 0 to 10 billion
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig('./figs/world_population_comparison.png', bbox_inches='tight')
    plt.close()


    print("All plots have been saved in the './figs' directory.")

if __name__ == "__main__":
    simulation_results = run_simulations()
    plot_results(simulation_results)