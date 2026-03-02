import { createContext, useContext, useReducer } from 'react';

const CartContext = createContext();

const cartReducer = (state, action) => {
    switch (action.type) {
        case 'ADD_ITEM': {
            const existing = state.items.find(
                (i) => i.id === action.payload.id && i.size === action.payload.size
            );
            if (existing) {
                return {
                    ...state,
                    items: state.items.map((i) =>
                        i.id === action.payload.id && i.size === action.payload.size
                            ? { ...i, quantity: i.quantity + 1 }
                            : i
                    ),
                };
            }
            return {
                ...state,
                items: [...state.items, { ...action.payload, quantity: 1 }],
            };
        }
        case 'REMOVE_ITEM':
            return {
                ...state,
                items: state.items.filter(
                    (i) => !(i.id === action.payload.id && i.size === action.payload.size)
                ),
            };
        case 'UPDATE_QUANTITY':
            return {
                ...state,
                items: state.items.map((i) =>
                    i.id === action.payload.id && i.size === action.payload.size
                        ? { ...i, quantity: Math.max(1, action.payload.quantity) }
                        : i
                ),
            };
        case 'CLEAR_CART':
            return { ...state, items: [] };
        default:
            return state;
    }
};

export function CartProvider({ children }) {
    const [state, dispatch] = useReducer(cartReducer, { items: [] });

    const addItem = (product, size) => {
        dispatch({
            type: 'ADD_ITEM',
            payload: { id: product.id, name: product.name, price: product.price, image: product.images[0], size },
        });
    };

    const removeItem = (id, size) => {
        dispatch({ type: 'REMOVE_ITEM', payload: { id, size } });
    };

    const updateQuantity = (id, size, quantity) => {
        dispatch({ type: 'UPDATE_QUANTITY', payload: { id, size, quantity } });
    };

    const clearCart = () => {
        dispatch({ type: 'CLEAR_CART' });
    };

    const totalItems = state.items.reduce((sum, i) => sum + i.quantity, 0);
    const totalPrice = state.items.reduce((sum, i) => sum + i.price * i.quantity, 0);

    return (
        <CartContext.Provider value={{ items: state.items, addItem, removeItem, updateQuantity, clearCart, totalItems, totalPrice }}>
            {children}
        </CartContext.Provider>
    );
}

export function useCart() {
    const context = useContext(CartContext);
    if (!context) throw new Error('useCart must be used within CartProvider');
    return context;
}
