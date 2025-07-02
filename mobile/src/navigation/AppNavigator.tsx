import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';

import HomeScreen from '../screens/HomeScreen';
import UploadScreen from '../screens/UploadScreen';
import HistoryScreen from '../screens/HistoryScreen';

export type RootTabParamList = {
  Home: undefined;
  Upload: undefined;
  History: undefined;
};

const Tab = createBottomTabNavigator<RootTabParamList>();

const AppNavigator: React.FC = () => {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ focused, color, size }) => {
            let iconName: keyof typeof Ionicons.glyphMap;

            if (route.name === 'Home') {
              iconName = focused ? 'home' : 'home-outline';
            } else if (route.name === 'Upload') {
              iconName = focused ? 'cloud-upload' : 'cloud-upload-outline';
            } else if (route.name === 'History') {
              iconName = focused ? 'list' : 'list-outline';
            } else {
              iconName = 'home-outline';
            }

            return <Ionicons name={iconName} size={size} color={color} />;
          },
          tabBarActiveTintColor: '#1e40af',
          tabBarInactiveTintColor: 'gray',
          headerStyle: {
            backgroundColor: '#1e40af',
          },
          headerTintColor: '#fff',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
        })}
      >
        <Tab.Screen 
          name="Home" 
          component={HomeScreen}
          options={{ title: 'MLB Pitch Recognition' }}
        />
        <Tab.Screen 
          name="Upload" 
          component={UploadScreen}
          options={{ title: 'Upload Video' }}
        />
        <Tab.Screen 
          name="History" 
          component={HistoryScreen}
          options={{ title: 'History' }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
};

export default AppNavigator;