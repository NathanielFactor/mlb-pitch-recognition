import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { StyleSheet, Platform } from 'react-native';

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
          tabBarActiveTintColor: '#007AFF',
          tabBarInactiveTintColor: 'rgba(0, 0, 0, 0.4)',
          tabBarStyle: {
            backgroundColor: 'rgba(255, 255, 255, 0.8)',
            borderTopWidth: 0,
            borderTopLeftRadius: 20,
            borderTopRightRadius: 20,
            paddingTop: 8,
            paddingBottom: Platform.OS === 'ios' ? 28 : 8,
            height: Platform.OS === 'ios' ? 88 : 64,
            position: 'absolute',
            shadowColor: 'rgba(0, 0, 0, 0.15)',
            shadowOffset: { width: 0, height: -4 },
            shadowOpacity: 1,
            shadowRadius: 16,
            elevation: 8,
          },
          tabBarLabelStyle: {
            fontSize: 12,
            fontWeight: '600',
            marginTop: 4,
          },
          headerShown: false,
        })}
      >
        <Tab.Screen 
          name="Home" 
          component={HomeScreen}
          options={{ title: 'Home' }}
        />
        <Tab.Screen 
          name="Upload" 
          component={UploadScreen}
          options={{ title: 'Upload' }}
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