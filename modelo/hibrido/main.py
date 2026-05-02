"""
Main - Punto de Entrada del Sistema Híbrido de Predicción

Este es el archivo principal para ejecutar predicciones de partidos
de la Premier League usando el sistema híbrido.
"""

import sys
import os

# Asegurar que el directorio hibrido está en el path
sys.path.insert(0, "/Users/mago-dios666@hotmail.com/modelo_predictivo/modelo/hibrido")

from hybrid_predict import predict_match, get_available_teams


def main():
    """Función principal para ejecutar predicciones."""
    
    print("\n" + "="*70)
    print("🏆 SISTEMA HÍBRIDO DE PREDICCIÓN - PREMIER LEAGUE")
    print("="*70)
    
    # Ejemplo de predicción
    print("\n📊 Realizando predicción de ejemplo...")
    print("-"*70)
    
    # Cambiar estos valores para predecir diferentes partidos
    home_team = "Arsenal FC"
    away_team = "Chelsea FC"
    
    # Realizar predicción
    resultado = predict_match(home_team, away_team)
    
    # El resultado ya se imprime dentro de predict_match
    # Aquí puedes agregar procesamiento adicional si lo necesitas
    
    if resultado:
        print("\n✅ Predicción completada exitosamente")
        print(f"Resultado guardado en variable: {type(resultado)}")
    else:
        print("\n❌ Error al realizar la predicción")
    
    return resultado


def prediccion_personalizada(home_team, away_team):
    """
    Realiza una predicción personalizada para equipos específicos.
    
    Args:
        home_team: Nombre del equipo local
        away_team: Nombre del equipo visitante
    
    Returns:
        dict con los resultados de la predicción
    """
    print("\n" + "="*70)
    print(f"🎯 PREDICCIÓN PERSONALIZADA")
    print("="*70)
    
    resultado = predict_match(home_team, away_team)
    return resultado


def mostrar_equipos_disponibles():
    """Muestra todos los equipos disponibles en el sistema."""
    print("\n" + "="*70)
    print("📋 EQUIPOS DISPONIBLES EN EL SISTEMA")
    print("="*70 + "\n")
    
    try:
        teams = get_available_teams()
        print(f"Total de equipos: {len(teams)}\n")
        
        for i, team in enumerate(teams, 1):
            print(f"{i:2d}. {team}")
        
        print("\n" + "="*70)
        return teams
    except Exception as e:
        print(f"❌ Error al obtener equipos: {e}")
        return []


def menu_interactivo():
    """Menú interactivo para usar el sistema."""
    
    while True:
        print("\n" + "="*70)
        print("🏆 MENÚ PRINCIPAL - SISTEMA DE PREDICCIÓN")
        print("="*70)
        print("\n1. Ver equipos disponibles")
        print("2. Realizar predicción personalizada")
        print("3. Predicción de ejemplo (Arsenal FC vs Chelsea FC)")
        print("4. Salir")
        print("\n" + "-"*70)
        
        opcion = input("Selecciona una opción (1-4): ").strip()
        
        if opcion == "1":
            mostrar_equipos_disponibles()
            
        elif opcion == "2":
            print("\n" + "-"*70)
            home = input("Equipo LOCAL: ").strip()
            away = input("Equipo VISITANTE: ").strip()
            
            if home and away:
                prediccion_personalizada(home, away)
            else:
                print("❌ Debes ingresar ambos equipos")
                
        elif opcion == "3":
            main()
            
        elif opcion == "4":
            print("\n👋 ¡Hasta luego!\n")
            break
            
        else:
            print("\n❌ Opción inválida. Por favor selecciona 1-4.")


# Ejemplos de uso directo
if __name__ == "__main__":
    # Opción 1: Ejecutar predicción de ejemplo
    #main()
    
    # Opción 2: Predicción personalizada (descomentar para usar)
    #prediccion_personalizada("Manchester United", "Liverpool")
    
    # Opción 3: Ver equipos disponibles (descomentar para usar)
    #mostrar_equipos_disponibles()
    
    # Opción 4: Menú interactivo (descomentar para usar)
    menu_interactivo()
