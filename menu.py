# main_menu.py
import subprocess

def main():
    while True:
        print("\n--------------------------------- MENÚ PRINCIPAL --------------------------------")
        print("1. Predecir complejidad algorítmica (Red 1)")
        print("2. Ordenar lista con BubbleSort NN (Red 2)")
        print("3. Simular operaciones de pila (Red 3) ")
        print("4. Simular operaciones de cola (Red 4)")
        print("5. Salir")

        opcion = input("Selecciona una opción (1-4): ")

        if opcion == '1':
            subprocess.run(["python", "red_1.py"])
        elif opcion == '2':
            subprocess.run(["python", "red_2.py"])
        elif opcion == '3':
            subprocess.run(["python", "red_3.py"])
        elif opcion == '4':
            subprocess.run(["python", "red_4.py"])
        elif opcion == '5':
            print("Saliendo del programa.")
            break
        else:
            print("Opción inválida. Intenta de nuevo.")

if __name__ == "__main__":
    main()
